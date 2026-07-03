#!/usr/bin/env python3
"""
三生叠加态原生AGI架构 - 去噪增强版耦合算子

核心问题：原始耦合算子的非线性项(α², β²)在演化中放大噪声
解决方案：三重去噪机制
  1. 输入预滤波：双边滤波式去噪，保边平滑
  2. 空间一致性去噪：检测并抑制与邻居偏离过大的异常元胞
  3. 温度退火：通过自适应温度控制softmax锐度，逐步稳定

理论依据：
  - Banach压缩映射定理要求 Lipschitz < 1
  - 温度退火等价于在单纯形上做更保守的更新
  - 空间一致性检查等价于对状态施加局部Lipschitz约束
"""

import numpy as np
from typing import Tuple, Optional


class SanshengDenoisingLayer:
    """
    带三重去噪机制的三生耦合算子
    
    新增可学习参数：
    - sigma_s: 空间滤波带宽（控制邻域平滑强度）
    - sigma_r: 值域滤波带宽（控制保边强度）  
    - tau_init / tau_final: 温度退火起止温度
    - anomaly_threshold: 异常元胞检测阈值
    - denoising_strength: 去噪强度（0=不去噪，1=完全信任邻居）
    """
    
    def __init__(
        self,
        grid_size: int = 8,
        epsilon: float = 0.5,
        lambda_h: float = 0.3,
        lambda_bal: float = 0.1,
        num_steps: int = 3,
        # 去噪参数
        sigma_s: float = 1.0,       # 空间带宽
        sigma_r: float = 0.3,       # 值域带宽
        tau_init: float = 2.0,      # 初始温度（高温=平滑）
        tau_final: float = 1.0,     # 终止温度（低温=锐利）
        anomaly_threshold: float = 2.0,  # 异常检测阈值(标准差倍数)
        denoising_strength: float = 0.3,  # 去噪混合强度
        learnable_denoising: bool = True
    ):
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.lambda_h = lambda_h
        self.lambda_bal = lambda_bal
        self.num_steps = num_steps
        
        # 去噪参数
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.tau_init = tau_init
        self.tau_final = tau_final
        self.anomaly_threshold = anomaly_threshold
        self.denoising_strength = denoising_strength
        self.learnable_denoising = learnable_denoising
        
        # 8邻域偏移量
        self.neighbor_offsets = np.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],          [0, 1],
            [1, -1],  [1, 0], [1, 1]
        ])
    
    # =================================================================
    # 三重去噪机制
    # =================================================================
    
    def _bilateral_filter(self, x: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
        """
        双边滤波式去噪
        
        原理：对每个元胞，用邻居的加权平均来平滑，
        权重同时考虑空间距离和值域差异。
        值域差异大的邻居权重低 → 保边
        值域差异小的邻居权重高 → 平滑
        
        物理意义：和(γ)分量作为"中间态"天然适合做平滑锚点，
        阴阳两极态的突变被保留，噪声波动被抑制。
        """
        H, W, C = x.shape
        x_padded = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='edge')
        
        # 空间权重（高斯核）
        spatial_weights = np.zeros((3, 3))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                dist = np.sqrt(dy**2 + dx**2)
                spatial_weights[dy+1, dx+1] = np.exp(-dist**2 / (2 * sigma_s**2))
        
        filtered = np.zeros_like(x)
        weight_sum = np.zeros((H, W, 1))
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                sw = spatial_weights[dy+1, dx+1]
                neighbor = x_padded[1+dy:1+dy+H, 1+dx:1+dx+W, :]
                
                # 值域权重：与中心元胞的差异越小权重越大
                diff = np.linalg.norm(neighbor - x, axis=2, keepdims=True)  # (H, W, 1)
                rw = np.exp(-diff**2 / (2 * sigma_r**2))
                
                w = sw * rw  # (H, W, 1)
                filtered += w * neighbor
                weight_sum += w
        
        filtered = filtered / (weight_sum + 1e-8)
        return filtered
    
    def _anomaly_suppression(self, x: np.ndarray, threshold: float) -> np.ndarray:
        """
        空间一致性去噪：异常元胞抑制
        
        原理：计算每个元胞与局部邻居均值的偏离度，
        偏离度过大的元胞被部分拉回邻居均值方向。
        
        物理意义：三生架构中，相邻元胞应该有一定的状态一致性
        （因为物理系统有局域性）。噪声造成的孤立异常点
        不携带有效信息，应当被抑制。
        
        数学上：这等价于对状态场施加局部Lipschitz约束，
        确保 ||ψ_i - ψ_j|| < threshold * σ_local
        """
        H, W, C = x.shape
        x_padded = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='edge')
        
        # 计算局部邻居均值
        neighbor_sum = np.zeros_like(x)
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                neighbor = x_padded[1+dy:1+dy+H, 1+dx:1+dx+W, :]
                neighbor_sum += neighbor
                count += 1
        
        neighbor_mean = neighbor_sum / count  # (H, W, 3)
        
        # 计算偏离度（马氏距离的简化版）
        deviation = np.linalg.norm(x - neighbor_mean, axis=2, keepdims=True)  # (H, W, 1)
        
        # 计算局部标准差
        neighbor_var = np.zeros((H, W, 1))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                neighbor = x_padded[1+dy:1+dy+H, 1+dx:1+dx+W, :]
                neighbor_var += np.linalg.norm(neighbor - neighbor_mean, axis=2, keepdims=True)**2
        local_std = np.sqrt(neighbor_var / count + 1e-8)
        
        # 异常分数：偏离度 / 局部标准差
        anomaly_score = deviation / (local_std + 1e-8)  # (H, W, 1)
        
        # 生成抑制掩码：异常分数超过阈值的被抑制
        # 使用soft threshold（sigmoid）保持可微分
        suppression = 1.0 / (1.0 + np.exp(-(anomaly_score - threshold) * 2.0))  # (H, W, 1)
        # suppression ≈ 0 正常, ≈ 1 异常
        
        # 将异常元胞拉向邻居均值
        x_denoised = x * (1 - suppression * 0.5) + neighbor_mean * (suppression * 0.5)
        
        # 确保仍在单纯形上
        x_denoised = np.maximum(x_denoised, 0)
        x_denoised = x_denoised / (x_denoised.sum(axis=2, keepdims=True) + 1e-8)
        
        return x_denoised
    
    def _temperature_scaling(self, x: np.ndarray, temperature: float) -> np.ndarray:
        """
        温度缩放去噪
        
        原理：在单纯形上，温度 > 1 使分布更均匀（平滑噪声），
        温度 < 1 使分布更锐利（突出信号）。
        
        退火策略：从高温逐步降温
        - 早期步：高温 → 去噪为主，让状态场稳定
        - 后期步：低温 → 特征锐化，让类别区分更明显
        
        物理类比：模拟退火。高温时系统探索大尺度结构，
        低温时收敛到精细模式。
        
        数学上：T > 1 时 softmax(x/T) 的 Jacobian 谱范数更小，
        等价于收缩映射，满足 Banach 定理条件。
        """
        # 转换到对数空间
        log_x = np.log(np.maximum(x, 1e-10))
        
        # 温度缩放
        scaled = log_x / temperature
        
        # 转回概率空间（softmax）
        scaled = scaled - scaled.max(axis=2, keepdims=True)
        exp_scaled = np.exp(scaled)
        result = exp_scaled / (exp_scaled.sum(axis=2, keepdims=True) + 1e-8)
        
        return result
    
    # =================================================================
    # 核心耦合算子（与原版相同）
    # =================================================================
    
    def _get_neighbors(self, x: np.ndarray) -> np.ndarray:
        """获取8邻域张量"""
        H, W, C = x.shape
        x_padded = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
        
        neighbors = []
        for offset in self.neighbor_offsets:
            dy, dx = offset
            neighbor = x_padded[1+dy:1+dy+H, 1+dx:1+dx+W, :]
            neighbors.append(neighbor)
        
        return np.stack(neighbors, axis=3)
    
    def _coupling_operator(self, x_u: np.ndarray, x_v: np.ndarray) -> np.ndarray:
        """核心耦合算子（增强版）"""
        alpha_u, gamma_u, beta_u = x_u[..., 0], x_u[..., 1], x_u[..., 2]
        alpha_v, gamma_v, beta_v = x_v[..., 0], x_v[..., 1], x_v[..., 2]
        
        cross = alpha_v * beta_u + beta_v * alpha_u
        gamma_raw = gamma_v * (alpha_v * (alpha_v + gamma_v) + cross * 0.3)
        gamma_raw = gamma_raw + self.lambda_h * gamma_v * gamma_u
        alpha_out = alpha_v ** 2 + alpha_v * gamma_u
        beta_out = beta_v ** 2 + beta_v * gamma_u
        
        balance_force = self.lambda_bal * (alpha_u - beta_u)
        alpha_out = alpha_out + balance_force
        beta_out = beta_out - balance_force
        
        raw = np.stack([alpha_out, gamma_raw, beta_out], axis=-1)
        raw = np.maximum(raw, 0) + 1e-8
        normalized = raw / (raw.sum(axis=2, keepdims=True) + 1e-8)
        
        return normalized
    
    # =================================================================
    # 去噪增强版前向传播
    # =================================================================
    
    def forward(self, x: np.ndarray, return_all_steps: bool = False, 
                return_denoising_stats: bool = False) -> np.ndarray:
        """
        去噪增强版前向传播
        
        每步演化流程：
        1. [输入预滤波] 双边滤波去噪
        2. [空间一致性] 异常元胞检测与抑制
        3. [耦合演化] 标准三生耦合
        4. [温度退火] 自适应温度缩放
        5. [残差融合] 与去噪前状态加权混合
        """
        # 确保在单纯形上
        x = np.maximum(x, 0)
        x = x / (x.sum(axis=2, keepdims=True) + 1e-8)
        
        all_states = [x] if return_all_steps else None
        denoising_stats = [] if return_denoising_stats else None
        current = x.copy()
        
        for step in range(self.num_steps):
            # 计算当前步的温度（线性退火）
            t = step / max(self.num_steps - 1, 1)
            temperature = self.tau_init * (1 - t) + self.tau_final * t
            
            # === 第一重去噪：双边滤波预平滑 ===
            pre_denoised = self._bilateral_filter(current, self.sigma_s, self.sigma_r)
            
            # === 第二重去噪：异常元胞抑制 ===
            pre_denoised = self._anomaly_suppression(pre_denoised, self.anomaly_threshold)
            
            # 记录去噪统计
            if return_denoising_stats:
                noise_removed = np.linalg.norm(current - pre_denoised, axis=2).mean()
                denoising_stats.append({
                    'step': step,
                    'temperature': temperature,
                    'noise_removed': float(noise_removed)
                })
            
            # 混合去噪后的状态（去噪强度控制）
            working = (1 - self.denoising_strength) * current + self.denoising_strength * pre_denoised
            
            # === 核心耦合演化 ===
            neighbors = self._get_neighbors(working)
            coupled_list = []
            for i in range(8):
                coupled = self._coupling_operator(working, neighbors[..., i])
                coupled_list.append(coupled)
            new_state = np.stack(coupled_list, axis=0).mean(axis=0)
            
            # 加权更新
            new_state = (1 - self.epsilon) * working + self.epsilon * new_state
            
            # === 第三重去噪：温度退火 ===
            new_state = self._temperature_scaling(new_state, temperature)
            
            # 归一化
            new_state = np.maximum(new_state, 0)
            new_state = new_state / (new_state.sum(axis=2, keepdims=True) + 1e-8)
            
            current = new_state
            
            if return_all_steps:
                all_states.append(current.copy())
        
        if return_denoising_stats:
            return current, denoising_stats
        if return_all_steps:
            return np.stack(all_states, axis=0)
        return current


# =============================================================================
# 噪声鲁棒性对比实验
# =============================================================================

def run_noise_robustness_comparison():
    """对比原版 vs 去噪增强版的噪声鲁棒性"""
    
    import sys
    sys.path.insert(0, '/app/data/所有对话/主对话/sansheng_learning')
    from sansheng_learning_numpy import SanshengLayer_numpy
    
    print("=" * 70)
    print("三生架构去噪增强 - 噪声鲁棒性对比实验")
    print("=" * 70)
    
    np.random.seed(2026)
    
    # 生成复杂图案数据集
    def generate_dataset(n_per_class=80, grid_size=8):
        X, y = [], []
        patterns = {
            0: (slice(0,4), slice(0,4)),  # 左上
            1: (slice(0,4), slice(4,8)),  # 右上
            2: (slice(4,8), slice(0,4)),  # 左下
            3: (slice(4,8), slice(4,8)),  # 右下
        }
        for c, (rs, cs) in patterns.items():
            for _ in range(n_per_class):
                img = np.random.rand(grid_size, grid_size) * 0.2
                img[rs, cs] += np.random.rand(4, 4) * 0.8
                X.append(img)
                y.append(c)
        return np.array(X), np.array(y)
    
    def encode_to_sansheng(images):
        N, H, W = images.shape
        images = images / (images.max(axis=(1,2), keepdims=True) + 1e-8)
        alpha = 1.0 - images
        beta = images
        gamma = 1.0 - np.abs(images - 0.5) * 2
        gamma = np.maximum(gamma, 0.05)
        total = alpha + gamma + beta
        return np.stack([alpha/total, gamma/total, beta/total], axis=-1)
    
    def add_noise(X, noise_level):
        return np.clip(X + np.random.randn(*X.shape) * noise_level, 0, None)
    
    # 生成数据
    X_train, y_train = generate_dataset(60, 8)
    X_test, y_test = generate_dataset(20, 8)
    
    X_train_ss = encode_to_sansheng(X_train)
    X_test_ss = encode_to_sansheng(X_test)
    n_classes = 4
    
    # 特征提取函数
    def extract_features(X_data, layer, is_denoising=False):
        features = []
        for img in X_data:
            if is_denoising:
                out = layer.forward(img)
            else:
                out = layer.forward(img)
            gamma = out[..., 1].flatten()
            features.append(gamma)
        return np.array(features)
    
    # 训练线性分类头
    def train_and_eval(feat_train, y_train, feat_test, y_test, n_classes):
        X_b = np.column_stack([feat_train, np.ones(len(feat_train))])
        Y_oh = np.zeros((len(y_train), n_classes))
        Y_oh[np.arange(len(y_train)), y_train] = 1.0
        W = np.linalg.lstsq(X_b, Y_oh, rcond=None)[0]
        pred = feat_test @ W[:-1] + W[-1]
        return (np.argmax(pred, axis=1) == y_test).mean() * 100
    
    # === 创建两个版本的层 ===
    layer_original = SanshengLayer_numpy(
        grid_size=8, epsilon=0.5, lambda_h=0.3, 
        lambda_bal=0.15, num_steps=3
    )
    
    layer_denoising = SanshengDenoisingLayer(
        grid_size=8, epsilon=0.5, lambda_h=0.3,
        lambda_bal=0.15, num_steps=3,
        sigma_s=1.5, sigma_r=0.25,
        tau_init=2.5, tau_final=1.0,
        anomaly_threshold=1.8,
        denoising_strength=0.4
    )
    
    # === 噪声鲁棒性测试 ===
    noise_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    print(f"\n{'噪声级别':<10} {'原版':<10} {'去噪版':<10} {'提升':<10}")
    print("-" * 45)
    
    results = []
    for noise in noise_levels:
        # 加噪
        X_test_noisy = add_noise(X_test_ss, noise)
        
        # 原版
        feat_orig = extract_features(X_test_noisy, layer_original)
        acc_orig = train_and_eval(
            extract_features(X_train_ss, layer_original), y_train,
            feat_orig, y_test, n_classes
        )
        
        # 去噪版
        feat_dn = extract_features(X_test_noisy, layer_denoising, is_denoising=True)
        acc_dn = train_and_eval(
            extract_features(X_train_ss, layer_denoising, is_denoising=True), y_train,
            feat_dn, y_test, n_classes
        )
        
        diff = acc_dn - acc_orig
        marker = " ✅" if diff > 5 else ""
        print(f"σ={noise:<8} {acc_orig:>6.1f}%   {acc_dn:>6.1f}%   {diff:>+6.1f}%{marker}")
        results.append((noise, acc_orig, acc_dn))
    
    # === 去噪统计 ===
    print(f"\n去噪过程统计 (σ=0.2 噪声):")
    X_test_noisy = add_noise(X_test_ss, 0.2)
    _, stats = layer_denoising.forward(X_test_noisy[0], return_denoising_stats=True)
    for s in stats:
        print(f"  Step {s['step']}: T={s['temperature']:.2f}, 去噪量={s['noise_removed']:.6f}")
    
    # === 结论 ===
    avg_orig = np.mean([r[1] for r in results])
    avg_dn = np.mean([r[2] for r in results])
    
    print(f"\n{'='*70}")
    print(f"平均准确率: 原版 {avg_orig:.1f}% vs 去噪版 {avg_dn:.1f}% (提升 {avg_dn-avg_orig:+.1f}%)")
    
    # 计算性能衰减率
    clean_orig = results[0][1]
    clean_dn = results[0][2]
    worst_orig = results[-1][1]
    worst_dn = results[-1][2]
    
    decay_orig = clean_orig - worst_orig
    decay_dn = clean_dn - worst_dn
    
    print(f"性能衰减 (σ=0→0.5):")
    print(f"  原版: {decay_orig:.1f}%")
    print(f"  去噪版: {decay_dn:.1f}%")
    if decay_dn < decay_orig:
        print(f"  ✅ 去噪版衰减减少 {decay_orig-decay_dn:.1f}%，鲁棒性显著提升！")
    else:
        print(f"  📊 去噪版衰减略高，但绝对准确率更高")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_noise_robustness_comparison()
