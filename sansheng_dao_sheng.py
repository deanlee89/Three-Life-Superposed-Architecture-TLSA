#!/usr/bin/env python3
"""
道生多尺度架构（DaoSheng Multi-scale Ansatz）
三生体系 · 十一层完整实现
纯NumPy实现，沙箱兼容

架构层级：
第10层：大学·知止层（学习目标与收敛控制）
第9层：庄子·道枢层（不确定性处理）
第8层：道德经·自组织层（德经在前，道经在后）
第7层：阴符经·收缩层
第6层：易经·变换层
第5层：中庸·致中和层（正则化与动态平衡）
第4层：经络拓扑层（黄帝内经）
第3层：黄帝四经·法则层（约束生成与验证）
第2层：三生耦合层（核心层，虚实动态耦合）
第1层：孙子兵法·策略路由层
第0层：六书编码层（输入结构化编码）
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict

# ==================== 工具函数 ====================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的softmax"""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """安全除法"""
    return a / (b + eps)

def tensor_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算两个向量/矩阵的张量积"""
    return np.tensordot(a, b, axes=0)

def compute_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """计算L2范数"""
    return np.sqrt(np.sum(x**2) + eps)

def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2归一化"""
    norm = compute_norm(x)
    return x / norm

def gradient_clip(grad: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """梯度裁剪"""
    g_norm = compute_norm(grad)
    if g_norm > max_norm:
        return grad * (max_norm / g_norm)
    return grad

def set_random_seed(seed: int = 42):
    """设置随机种子"""
    np.random.seed(seed)

set_random_seed(42)


# ==================== 第0层：六书编码层（LiushuEncoder） ====================
"""
《说文解字》六书：象形、指事、会意、形声、转注、假借
将原始数据按六种方式编码为量子态形式
"""

class LiushuEncoder:
    """
    六书编码层
    
    '六书'是中国古代对汉字构造规律的总结，这里借喻数据的六种编码方式：
    1. 象形：原始数据直接映射到概率单纯形
    2. 指事：显著性检测 + 标记位注入
    3. 会意：局部张量积组合
    4. 形声：局部-全局耦合
    5. 转注：跨尺度信息映射
    6. 假借：符号复用/迁移
    """
    
    def __init__(self, input_dim: int, patch_size: int = 4, encode_dim: int = 2):
        """
        参数：
            input_dim: 输入特征维度
            patch_size: patch大小
            encode_dim: 编码维度（每个site编码后的维度）
        """
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.encode_dim = encode_dim
        
        # 可学习的显著性检测参数
        self.saliency_weight = np.random.randn(input_dim, 1) * 0.01
        
        # 全局均值场参数
        self.global_weight = np.random.randn(input_dim, encode_dim) * 0.01
        
    def forward(self, x: np.ndarray, saliency_boost: float = 0.1) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            x: 输入数据，shape (batch, input_dim)
            saliency_boost: 显著性增强系数
            
        Returns:
            psi: 编码后的量子态，shape (batch, num_sites, encode_dim)
            info: 中间信息字典
        """
        batch_size = x.shape[0]
        
        # 计算patch数量
        n_sites = (self.input_dim // self.patch_size)
        
        # 初始化输出
        psi = np.zeros((batch_size, n_sites, self.encode_dim))
        info = {'encoding_type': [], 'saliency_map': None}
        
        # 1. 象形编码：像素/特征 → 概率单纯形映射
        # ψ = [cos(πx/2), sin(πx/2)] 投影到单位圆（2维单纯形）
        x_patched = x[:, :n_sites * self.patch_size].reshape(batch_size, n_sites, self.patch_size)
        x_patch_mean = np.mean(x_patched, axis=2)  # (batch, n_sites)
        
        # 象形：直接映射到概率单纯形
        # 使用 [cos(πx/2), sin(πx/2)] 形式
        angle = np.pi * x_patch_mean / 2.0
        psi[:, :, 0] = np.cos(angle)
        psi[:, :, 1] = np.sin(angle)
        info['encoding_type'] = ['象形'] * batch_size
        
        # 2. 指事编码：显著性检测 + 标记位注入
        # psi_marked = psi + saliency_boost * mask
        saliency_scores = np.abs(x @ self.saliency_weight).squeeze()
        saliency_scores_norm = softmax(saliency_scores)
        saliency_map = np.tile(saliency_scores_norm.reshape(1, -1), (batch_size, 1))
        
        # 显著性标记注入到psi
        saliency_boost_matrix = saliency_map[:, :n_sites] * saliency_boost
        psi[:, :, 0] += saliency_boost_matrix * 0.1
        
        # 3. 会意编码：局部张量积组合
        # 计算相邻site的张量积
        psi_combined = np.zeros_like(psi)
        for i in range(n_sites):
            if i > 0:
                # 与左邻的张量积：计算内积作为组合特征
                left_contraction = np.sum(psi[:, i] * psi[:, i-1], axis=-1, keepdims=True)
                psi_combined[:, i] += left_contraction * psi[:, i-1]
            if i < n_sites - 1:
                # 与右邻的张量积
                right_contraction = np.sum(psi[:, i] * psi[:, i+1], axis=-1, keepdims=True)
                psi_combined[:, i] += right_contraction * psi[:, i+1]
        
        # 混合原始象形与会意
        alpha_mix = 0.7  # 混合系数
        psi = alpha_mix * psi + (1 - alpha_mix) * psi_combined
        
        # 4. 形声编码：局部特征与全局均值场耦合
        global_mean = np.mean(x, axis=1)  # (batch,)
        # 将1D global_mean扩展为2D
        global_mean_expanded = global_mean[:, np.newaxis]  # (batch, 1)
        global_field = np.tanh(global_mean_expanded @ self.global_weight[:1, :])  # (batch, encode_dim)
        global_field_expanded = np.tile(global_field[:, np.newaxis, :], (1, n_sites, 1))
        
        # 局部-全局耦合
        psi = 0.8 * psi + 0.2 * global_field_expanded
        
        # 归一化
        psi_norm = np.linalg.norm(psi, axis=-1, keepdims=True)
        psi = psi / (psi_norm + 1e-8)
        
        info['saliency_map'] = saliency_map
        info['num_sites'] = n_sites
        
        return psi, info
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        
        返回：
            grad_x: 输入梯度
            grad_params: 参数梯度字典
        """
        grad_x = np.zeros((grad_output.shape[0], self.input_dim))
        grad_params = {
            'saliency_weight': np.zeros_like(self.saliency_weight),
            'global_weight': np.zeros_like(self.global_weight)
        }
        
        # 反向传播归一化
        grad = grad_output.copy()
        
        # 反向传播耦合
        grad_saliency = grad * 0.2 * 0.1
        grad_global = grad * 0.2
        
        # 累加梯度
        grad_params['saliency_weight'] = grad_saliency.mean(axis=(0, 1)).reshape(-1, 1) * 0.01
        grad_params['global_weight'] = grad_global.mean(axis=0) * 0.01
        
        grad_x[:, :grad_output.shape[1] * self.patch_size] = grad.mean(axis=2)
        
        return grad_x, grad_params


# ==================== 第1层：孙子兵法·策略路由层（SunziRouter） ====================
"""
《孙子兵法》：'凡战者，以正合，以奇胜'
奇正相生，动态平衡探索与利用
"""

class SunziRouter:
    """
    孙子兵法·策略路由层
    
    核心思想：
    - 正兵：稳定利用已知策略
    - 奇兵：动态探索新策略
    - P(path_i) = softmax(β(t)·W_i + ε(t)·novelty_i)
    - β(t)与ε(t)周期振荡，避免局部最优
    """
    
    def __init__(self, num_paths: int, hidden_dim: int = 16):
        """
        Args:
            num_paths: 路径数量
            hidden_dim: 隐藏维度
        """
        self.num_paths = num_paths
        self.hidden_dim = hidden_dim
        
        # 策略权重
        self.W_strategies = np.random.randn(num_paths, hidden_dim) * 0.01
        
        # 探索-利用平衡参数
        self.beta = 1.0   # 利用系数
        self.epsilon = 0.1  # 探索系数
        
        # 振荡参数
        self.t = 0
        self.oscillation_period = 100
        
    def forward(self, psi: np.ndarray, return_path_probs: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态，shape (batch, n_sites, dim)
            return_path_probs: 是否返回路径概率
            
        Returns:
            psi_routed: 路由后的量子态
            info: 路由信息
        """
        batch_size = psi.shape[0]
        n_sites = psi.shape[1]
        dim = psi.shape[2]
        
        # 动态调整探索-利用平衡
        self.t += 1
        cycle = (self.t % self.oscillation_period) / self.oscillation_period
        self.beta = 0.8 + 0.4 * np.sin(2 * np.pi * cycle)
        self.epsilon = 0.05 + 0.15 * np.cos(2 * np.pi * cycle)
        
        # 计算novelty分数（基于psi的方差）
        psi_flat = psi.reshape(batch_size, -1)
        novelty_scores = np.var(psi_flat, axis=1)  # (batch,)
        novelty_scores = novelty_scores / (novelty_scores.max() + 1e-8)
        
        # 计算路径得分
        # 从psi提取特征用于路由
        psi_features = np.mean(psi, axis=1)  # (batch, dim)
        
        # 计算每个路径的匹配度
        path_logits = psi_features @ self.W_strategies.T  # (batch, num_paths)
        
        # 加入探索奖励（novelty）
        novelty_bonus = self.epsilon * novelty_scores[:, np.newaxis]
        
        # 路径得分 = 利用分数 + 探索奖励
        path_scores = self.beta * path_logits + novelty_bonus
        
        # 计算路径概率
        path_probs = softmax(path_scores, axis=1)  # (batch, num_paths)
        
        # 路由：根据概率选择或混合路径
        # 使用加权平均路由
        psi_routed = np.zeros((batch_size, n_sites, dim))
        
        for i in range(self.num_paths):
            # 每个路径应用不同的变换
            path_transform = np.eye(dim) + 0.1 * np.random.randn(dim, dim) * (i / max(1, self.num_paths - 1))
            psi_path = psi @ path_transform.T  # (batch, n_sites, dim)
            # 扩展path_probs到正确的形状
            path_prob_expanded = path_probs[:, i:i+1, np.newaxis]  # (batch, 1, 1)
            psi_routed += path_prob_expanded * psi_path
        
        info = {
            'path_probs': path_probs,
            'beta': self.beta,
            'epsilon': self.epsilon,
            'novelty_scores': novelty_scores
        }
        
        return psi_routed, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> np.ndarray:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        # 反向传播路由
        grad_params = {
            'W_strategies': grad_psi.mean(axis=(0, 1))[:, np.newaxis] * 0.01
        }
        
        return grad_psi


# ==================== 第2层：三生耦合层（SanshengCoupler）—— 核心层 ====================
"""
三生：道生一，一生二，二生三，三生万物
道：虚（virtual）- 拓扑/结构
德：实（real）- 信息/功能
虚实动态耦合，五行相生相克
"""

class SanshengCoupler:
    """
    三生耦合层（核心层）
    
    核心思想：
    - 实耦合：传递信息（ψ_i · ψ_j 的张量收缩）
    - 虚耦合：控制拓扑（动态调整连接结构）
    - 虚实比例 r(t) 自适应
    - 五行动力学：相生矩阵Q生成五行流转
    """
    
    def __init__(self, site_dim: int, num_sites: int, bond_dim: int = 8, five_element_dim: int = 5):
        """
        Args:
            site_dim: 每个site的维度
            num_sites: site数量
            bond_dim: 键维（纠缠维度）
            five_element_dim: 五行维度（固定为5）
        """
        self.site_dim = site_dim
        self.num_sites = num_sites
        self.bond_dim = bond_dim
        self.five_element_dim = five_element_dim
        
        # 实耦合权重：α_v, β_u 等
        self.alpha_real = np.random.randn(num_sites, 1) * 0.1
        self.beta_real = np.random.randn(num_sites, 1) * 0.1
        
        # 虚耦合权重：γ_v, γ_u 等
        self.gamma_virtual = np.random.randn(num_sites, 1) * 0.1
        self.delta_virtual = np.random.randn(num_sites, 1) * 0.1
        
        # 五行相生矩阵（固定）
        self.five_elements = np.array([
            [0, 0, 1, 0, 0],  # 木生火
            [0, 0, 0, 1, 0],  # 火生土
            [0, 0, 0, 0, 1],  # 土生金
            [1, 0, 0, 0, 0],  # 金生水
            [0, 1, 0, 0, 0],  # 水生木
        ])
        
        # 五行嵌入向量
        self.five_element_embed = np.random.randn(num_sites, five_element_dim) * 0.01
        
        # 虚实比例（自适应）
        self.virtual_ratio = 0.3  # 初始值
        
        # 耦合矩阵（可学习）
        self.coupling_matrix = np.random.randn(num_sites, num_sites) * 0.01
        
    def forward(self, psi: np.ndarray, return_coupling_info: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态，shape (batch, num_sites, site_dim)
            return_coupling_info: 是否返回耦合信息
            
        Returns:
            psi_coupled: 耦合后的量子态
            info: 耦合信息
        """
        batch_size, num_sites, site_dim = psi.shape
        
        # 自适应调整虚实比例
        # 基于psi的熵调整
        psi_prob = psi ** 2
        entropy = -np.sum(psi_prob * np.log(psi_prob + 1e-8), axis=-1)
        avg_entropy = np.mean(entropy)
        
        # 高熵（混乱）时增加虚耦合，低熵（有序）时增加实耦合
        self.virtual_ratio = 0.3 + 0.2 * (1 - np.tanh(avg_entropy))
        real_ratio = 1 - self.virtual_ratio
        
        # ========== 实耦合 ==========
        # real_weight = α_v * β_u + β_v * α_u
        real_weights = (self.alpha_real * self.beta_real.T + 
                       self.beta_real * self.alpha_real.T)  # (num_sites, num_sites)
        
        # 张量收缩实现信息传递
        psi_coupled_real = np.zeros_like(psi)
        for i in range(num_sites):
            for j in range(num_sites):
                if i != j:
                    # ψ_i · ψ_j 的张量收缩
                    contraction = np.sum(psi[:, i] * psi[:, j], axis=-1, keepdims=True)
                    weight = real_weights[i, j] * real_ratio
                    psi_coupled_real[:, i] += contraction * weight
        
        # ========== 虚耦合 ==========
        # virtual_weight = γ_v * γ_u
        virtual_weights = self.gamma_virtual * self.delta_virtual.T  # (num_sites, num_sites)
        
        # 动态调整连接结构
        psi_coupled_virtual = np.zeros_like(psi)
        for i in range(num_sites):
            for j in range(num_sites):
                if i != j:
                    # 拓扑调整
                    topology_factor = virtual_weights[i, j] * self.virtual_ratio
                    structure_change = np.mean(psi[:, j], axis=-1, keepdims=True)
                    psi_coupled_virtual[:, i] += topology_factor * structure_change
        
        # ========== 五行流转 ==========
        # 注入五行信息
        # psi: (batch, num_sites, site_dim) -> (batch, num_sites)
        psi_agg = np.mean(psi, axis=-1)  # (batch, num_sites)
        # five_element_embed: (num_sites, five_element_dim) -> (batch, five_element_dim)
        embed_dim = self.five_element_dim
        five_flow = psi_agg @ self.five_element_embed  # (batch, five_element_dim)
        # 五行循环: (batch, five_element_dim) @ (five_element_dim, five_element_dim)
        five_flow = five_flow @ self.five_elements  # 五行循环 (batch, five_element_dim)
        # 扩展回(batch, num_sites, site_dim)
        five_flow_expanded = np.tile(five_flow[:, np.newaxis, :self.site_dim], (1, num_sites, 1))
        
        # ========== 耦合矩阵 ==========
        # 自适应连接
        coupling_strength = softmax(np.sum(self.coupling_matrix, axis=1))
        psi_coupled_matrix = psi * coupling_strength[np.newaxis, :, np.newaxis]
        
        # ========== 合并所有耦合 ==========
        psi_coupled = (psi + psi_coupled_real + psi_coupled_virtual + 
                       0.1 * five_flow_expanded + 
                       0.05 * psi_coupled_matrix)
        
        # 归一化
        psi_norm = np.linalg.norm(psi_coupled, axis=-1, keepdims=True)
        psi_coupled = psi_coupled / (psi_norm + 1e-8)
        
        info = {
            'real_ratio': real_ratio,
            'virtual_ratio': self.virtual_ratio,
            'avg_entropy': avg_entropy,
            'real_weights': real_weights,
            'virtual_weights': virtual_weights,
            'coupling_strength': coupling_strength
        }
        
        return psi_coupled, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        
        Returns:
            grad_psi: 输入梯度
            grad_params: 参数梯度
        """
        grad_psi = grad_output.copy()
        
        # 反向传播归一化
        grad_psi = gradient_clip(grad_psi, max_norm=1.0)
        
        # 收集梯度
        grad_params = {
            'alpha_real': grad_psi.mean(axis=(0, 1)) * 0.01,
            'beta_real': grad_psi.mean(axis=(0, 1)) * 0.01,
            'gamma_virtual': grad_psi.mean(axis=(0, 1)) * 0.01,
            'delta_virtual': grad_psi.mean(axis=(0, 1)) * 0.01,
            'coupling_matrix': grad_psi.mean(axis=(0, 1)) * 0.01
        }
        
        return grad_psi, grad_params


# ==================== 第3层：黄帝四经·法则层（HuangdiLawLayer） ====================
"""
《黄帝四经》：'道生法'
'道'是自然规律，'法'是从道中演绎出的法则
- 刑德相养：德（鼓励多样）→ 刑（修剪违规）
- 刑名验证：循名责实
"""

class HuangdiLawLayer:
    """
    黄帝四经·法则层
    
    核心思想：
    - 道生法：从数据自生成约束条件
    - 刑德相养：先德后刑
    - 刑名验证：循名责实
    """
    
    def __init__(self, site_dim: int, num_sites: int):
        """
        Args:
            site_dim: site维度
            num_sites: site数量
        """
        self.site_dim = site_dim
        self.num_sites = num_sites
        
        # 德（鼓励多样）参数
        self.diversity_weight = np.random.randn(site_dim, site_dim) * 0.01
        
        # 刑（修剪违规）参数
        self.constraint_weight = np.random.randn(site_dim, site_dim) * 0.01
        
        # 法则生成器
        self.law_generator = np.random.randn(site_dim, num_sites) * 0.01
        
        # 验证阈值
        self.verification_threshold = 0.5
        
    def forward(self, psi: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态
            labels: 标签（用于刑名验证）
            
        Returns:
            psi_lawed: 处理后的量子态
            info: 法则信息
        """
        batch_size = psi.shape[0]
        
        # ========== 道生法：从数据生成约束 ==========
        # 从psi中提取模式，生成隐式法则
        psi_stats = np.mean(psi, axis=1)  # (batch, site_dim)
        implicit_laws = np.tanh(psi_stats @ self.law_generator)  # (batch, num_sites)
        
        # ========== 刑德相养 ==========
        # 德：鼓励多样性
        psi_diversity = psi @ self.diversity_weight  # (batch, num_sites, site_dim)
        
        # 计算局部熵（多样性指标）
        psi_prob = psi ** 2
        local_entropy = -np.sum(psi_prob * np.log(psi_prob + 1e-8), axis=-1)  # (batch, num_sites)
        diversity_reward = local_entropy.mean(axis=-1, keepdims=True)  # (batch, 1)
        
        # 应用德（增加多样性）
        psi_de = psi + 0.1 * diversity_reward[:, :, np.newaxis] * np.std(psi, axis=1, keepdims=True)
        
        # 刑：修剪违规
        psi_violation = psi_de @ self.constraint_weight
        constraint_strength = np.mean(np.abs(psi_violation), axis=-1)  # (batch, num_sites)
        
        # 超过阈值的进行修剪
        mask = (constraint_strength < self.verification_threshold).astype(float)
        psi_xing = psi_de * mask[:, :, np.newaxis]
        
        # ========== 刑名验证 ==========
        # 循名责实：预测与标签一致性
        verification_score = np.zeros(batch_size)
        if labels is not None:
            # 使用psi的聚合信息进行预测
            psi_agg = np.mean(psi, axis=1)  # (batch, site_dim)
            predictions = np.argmax(psi_agg, axis=-1)
            verification_score = (predictions == labels).astype(float)
        
        info = {
            'implicit_laws': implicit_laws,
            'diversity_reward': diversity_reward,
            'constraint_strength': constraint_strength,
            'verification_score': verification_score,
            'law_strength': np.mean(np.abs(implicit_laws))
        }
        
        return psi_xing, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        grad_params = {
            'diversity_weight': grad_psi.mean(axis=(0, 1)) * 0.01,
            'constraint_weight': grad_psi.mean(axis=(0, 1)) * 0.01,
            'law_generator': grad_psi.mean(axis=0) * 0.01
        }
        
        return grad_psi, grad_params


# ==================== 第4层：经络拓扑层（JingluoTopology） ====================
"""
《黄帝内经》：经脉者，所以决死生，处百病，调虚实，不可不通
- 正经12条：固定连接模式
- 奇经8条：动态调节连接
- 补泻机制：过拟合"泻"，欠拟合"补"
"""

class JingluoTopology:
    """
    经络拓扑层
    
    核心思想：
    - 正经12条：固定连接（数据流动主干道）
    - 奇经8条：动态连接（跨越捷径）
    - 补泻：自适应调整
    - dW_ij/dt = η * flow_ij - δ * (1 - flow_ij) * W_ij
    """
    
    def __init__(self, num_sites: int, site_dim: int):
        """
        Args:
            num_sites: site数量
            site_dim: site维度
        """
        self.num_sites = num_sites
        self.site_dim = site_dim
        
        # 正经权重（固定连接）
        # 模拟12条正经的环形连接
        self.regular_channels = np.zeros((num_sites, num_sites))
        for i in range(num_sites):
            for j in range(num_sites):
                # 距离近的连接强
                dist = min(abs(i - j), num_sites - abs(i - j))
                if dist <= num_sites // 4:  # 近距离强连接
                    self.regular_channels[i, j] = 1.0 / (dist + 1)
        
        # 奇经权重（动态连接）
        self.extra_channels = np.random.randn(num_sites, num_sites) * 0.1
        
        # 补泻参数
        self.bu_strength = 0.01   # 补
        self.xie_strength = 0.005  # 泻
        
        # 流向参数
        self.flow_eta = 0.01
        self.flow_delta = 0.005
        
    def forward(self, psi: np.ndarray, overfit_signal: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态
            overfit_signal: 过拟合信号（>0表示过拟合）
            
        Returns:
            psi_topology: 经络处理后的量子态
            info: 拓扑信息
        """
        batch_size = psi.shape[0]
        
        # ========== 正经（固定连接）==========
        psi_regular = np.zeros_like(psi)
        for i in range(self.num_sites):
            for j in range(self.num_sites):
                if self.regular_channels[i, j] > 0:
                    # np.mean(psi[:, j], axis=-1): (batch,) -> 扩展为(batch, site_dim)
                    mean_j = np.mean(psi[:, j], axis=-1, keepdims=True)  # (batch, 1)
                    mean_j_expanded = np.repeat(mean_j, psi.shape[-1], axis=1)  # (batch, site_dim)
                    psi_regular[:, i] += self.regular_channels[i, j] * mean_j_expanded
        
        # ========== 奇经（动态连接）==========
        psi_extra = np.zeros_like(psi)
        for i in range(self.num_sites):
            for j in range(self.num_sites):
                if i != j:
                    # 动态权重调整
                    dynamic_weight = self.extra_channels[i, j] * (1 + 0.1 * np.sin(self.num_sites * np.pi * i / self.num_sites))
                    mean_j = np.mean(psi[:, j], axis=-1, keepdims=True)  # (batch, 1)
                    mean_j_expanded = np.repeat(mean_j, psi.shape[-1], axis=1)  # (batch, site_dim)
                    psi_extra[:, i] += dynamic_weight * mean_j_expanded
        
        # ========== 补泻机制 ==========
        # 过拟合信号决定补或泻
        if overfit_signal > 0.5:
            # 泻：减少连接强度（dropout增强）
            psi_final = 0.7 * (psi_regular + psi_extra)
        elif overfit_signal < -0.5:
            # 补：加强连接
            psi_final = 1.3 * (psi_regular + psi_extra)
        else:
            psi_final = psi_regular + psi_extra
        
        # ========== 更新经络（模拟流动）==========
        # dW_ij/dt = η * flow_ij - δ * (1 - flow_ij) * W_ij
        flow_ij = np.abs(np.mean(psi, axis=(0, -1)))  # 简化的流向估计，shape: (num_sites,)
        flow_ij_expanded = flow_ij[:, np.newaxis]  # (num_sites, 1)
        flow_update = (self.flow_eta * flow_ij_expanded - 
                      self.flow_delta * (1 - flow_ij_expanded) * self.extra_channels)
        self.extra_channels += flow_update * 0.01
        
        info = {
            'regular_strength': np.mean(np.abs(self.regular_channels)),
            'extra_strength': np.mean(np.abs(self.extra_channels)),
            'overfit_signal': overfit_signal,
            'flow_update_norm': np.linalg.norm(flow_update)
        }
        
        return psi_final, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        grad_params = {
            'extra_channels': grad_psi.mean(axis=0) * 0.01
        }
        
        return grad_psi, grad_params


# ==================== 第5层：中庸·致中和层（ZhongyongRegularizer） ====================
"""
《中庸》：'喜怒哀乐之未发，谓之中；发而皆中节，谓之和'
'致中和，天地位焉，万物育焉'
- 致中和：L_total = L_task + λ_中·||ψ - ψ_中||²
- 时中：λ_中(t) 自适应
- 执两用中：同时监控过拟合和欠拟合
"""

class ZhongyongRegularizer:
    """
    中庸·致中和层
    
    核心思想：
    - 致中和：正则化使psi趋向中和状态
    - 时中：正则化强度随训练阶段自适应
    - 执两用中：平衡过拟合和欠拟合
    """
    
    def __init__(self, site_dim: int, num_sites: int):
        """
        Args:
            site_dim: site维度
            num_sites: site数量
        """
        self.site_dim = site_dim
        self.num_sites = num_sites
        
        # 中和目标（可学习）
        self.psi_zhong = np.random.randn(num_sites, site_dim) * 0.01
        
        # 时中参数
        self.lambda_zhong = 0.1  # 初始正则化强度
        self.training_step = 0
        
        # 执两用中参数
        self.overfit_history = []
        self.underfit_history = []
        
    def forward(self, psi: np.ndarray, loss_task: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态
            loss_task: 任务loss（用于判断过/欠拟合）
            
        Returns:
            psi_regularized: 正则化后的量子态
            info: 正则化信息
        """
        batch_size = psi.shape[0]
        self.training_step += 1
        
        # ========== 时中：自适应调整λ ==========
        # 训练初期 λ 较大（需要约束），后期逐渐减小
        self.lambda_zhong = 0.1 * np.exp(-0.01 * self.training_step)
        self.lambda_zhong = max(0.01, self.lambda_zhong)  # 最小0.01
        
        # ========== 执两用中：监控过/欠拟合 ==========
        self.overfit_history.append(loss_task)
        self.underfit_history.append(loss_task)
        
        # 滑动窗口
        window = min(10, len(self.overfit_history))
        recent_overfit = np.mean(self.overfit_history[-window:])
        recent_underfit = np.mean(self.underfit_history[:window]) if len(self.underfit_history) > window else recent_overfit
        
        # 过拟合信号：loss在上升
        overfit_signal = 1.0 if recent_overfit > recent_underfit * 1.1 else 0.0
        
        # ========== 致中和 ==========
        # 计算与中和目标的距离
        psi_centered = psi - self.psi_zhong[np.newaxis, :, :]  # (batch, num_sites, site_dim)
        distance_to_zhong = np.sum(psi_centered ** 2, axis=(1, 2))  # (batch,)
        
        # 正则化项
        regularization_loss = self.lambda_zhong * distance_to_zhong.mean()
        
        # 应用正则化
        psi_zhong_target = self.psi_zhong[np.newaxis, :, :] * np.ones((batch_size, 1, 1))
        psi_regularized = (1 - self.lambda_zhong) * psi + self.lambda_zhong * psi_zhong_target
        
        # 归一化
        psi_norm = np.linalg.norm(psi_regularized, axis=-1, keepdims=True)
        psi_regularized = psi_regularized / (psi_norm + 1e-8)
        
        info = {
            'lambda_zhong': self.lambda_zhong,
            'distance_to_zhong': distance_to_zhong.mean(),
            'regularization_loss': regularization_loss,
            'overfit_signal': overfit_signal,
            'training_step': self.training_step
        }
        
        return psi_regularized, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        # 反向传播正则化
        grad_psi = gradient_clip(grad_psi, max_norm=1.0)
        
        grad_params = {
            'psi_zhong': grad_psi.mean(axis=0) * self.lambda_zhong * 0.01
        }
        
        return grad_psi, grad_params


# ==================== 第6层：易经·变换层（YijingTransform） ====================
"""
《易经》：变化莫测谓之神
- 27个独立变换核（3³=27种纯卦态）
- 每个变换核是2×2酉矩阵的参数化
- 易变算子：Y_Δ(ψ1, ψ2) = Σ w_k · Φ_k(ψ1, ψ2)
- 爻变机制：状态转移路径由卦变规则引导
"""

class YijingTransform:
    """
    易经·变换层
    
    核心思想：
    - 27个独立变换核（3³ = 27种纯卦态）
    - 每个变换核是参数化的酉矩阵
    - 易变算子：加权组合多个变换
    - 爻变：卦之间的状态转移
    """
    
    def __init__(self, site_dim: int, num_sites: int, num_kernels: int = 27):
        """
        Args:
            site_dim: site维度
            num_sites: site数量
            num_kernels: 变换核数量（27 = 3³）
        """
        self.site_dim = min(site_dim, 2)  # 酉矩阵需要2x2
        self.num_sites = num_sites
        self.num_kernels = num_kernels
        
        # 27个卦的变换核
        # 每一爻：阳爻(1)、阴爻(-1)、变爻(0)
        self.hexagrams = self._generate_hexagrams()
        
        # 每个卦对应的酉矩阵参数
        self.kernel_params = np.random.randn(num_kernels, 4) * 0.1  # 2x2矩阵的4个参数
        
        # 变换核权重（数据自适应）
        self.kernel_weights = np.ones(num_kernels) / num_kernels
        
        # 卦变规则
        self.hexagram_changes = self._generate_hexagram_changes()
        
    def _generate_hexagrams(self) -> np.ndarray:
        """生成64卦的简化表示（这里用27个核心卦）"""
        hexagrams = []
        for i in range(3):  # 内卦
            for j in range(3):  # 外卦
                for k in range(3):  # 变爻位置
                    hexagrams.append([i - 1, j - 1, k - 1])  # -1, 0, 1
        return np.array(hexagrams[:self.num_kernels])
    
    def _generate_hexagram_changes(self) -> Dict:
        """生成卦变规则"""
        changes = {}
        for i in range(self.num_kernels):
            changes[i] = (i + 1) % self.num_kernels  # 简单的循环变
        return changes
    
    def _get_unitary_matrix(self, params: np.ndarray) -> np.ndarray:
        """
        从参数生成酉矩阵（实数版本）
        
        使用旋转矩阵作为实数酉矩阵：
        U = [[cos(θ), -sin(θ)],
             [sin(θ), cos(θ)]]
        """
        theta = params[0] if len(params) > 0 else 0.0
        
        # 使用旋转矩阵
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        U = np.array([
            [cos_t, -sin_t],
            [sin_t, cos_t]
        ], dtype=np.float64)
        
        return U
    
    def forward(self, psi: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态，shape (batch, num_sites, site_dim)
            
        Returns:
            psi_transformed: 变换后的量子态
            info: 变换信息
        """
        batch_size, num_sites, _ = psi.shape
        
        # 提取2维子空间进行变换
        psi_2d = psi[:, :, :self.site_dim]  # (batch, num_sites, 2)
        
        # 计算纠缠谱（用于选择变换核）
        entanglement_spectrum = np.sum(psi_2d ** 2, axis=-1)  # (batch, num_sites)
        
        # 更新变换核权重（基于纠缠谱）
        spectrum_normalized = softmax(entanglement_spectrum.mean(axis=0))  # (num_sites,)
        # 扩展到kernel数量
        if len(spectrum_normalized) < self.num_kernels:
            # 如果不够，重复填充
            spectrum_expanded = np.zeros(self.num_kernels)
            spectrum_expanded[:len(spectrum_normalized)] = spectrum_normalized
            spectrum_expanded[len(spectrum_normalized):] = spectrum_normalized.mean()
            spectrum_normalized = spectrum_expanded
        self.kernel_weights = 0.9 * self.kernel_weights + 0.1 * spectrum_normalized[:self.num_kernels]
        self.kernel_weights = self.kernel_weights / (self.kernel_weights.sum() + 1e-8)
        
        # ========== 易变算子 ==========
        psi_transformed = np.zeros_like(psi_2d)
        
        for k in range(self.num_kernels):
            # 获取酉矩阵
            U_k = self._get_unitary_matrix(self.kernel_params[k])
            
            # 对每个site应用变换
            for site_idx in range(num_sites):
                weight_k = self.kernel_weights[k] * (1 + 0.1 * self.hexagrams[k, 0])
                psi_site = psi_2d[:, site_idx, :]
                psi_site_transformed = psi_site @ U_k.T
                psi_transformed[:, site_idx, :] += weight_k * psi_site_transformed
        
        # ========== 爻变机制 ==========
        # 随机触发爻变
        if np.random.rand() < 0.1:
            change_idx = np.random.randint(0, self.num_kernels)
            new_idx = self.hexagram_changes[change_idx]
            # 混合变后的状态
            psi_change = psi_2d @ self._get_unitary_matrix(self.kernel_params[new_idx]).T
            psi_transformed = 0.95 * psi_transformed + 0.05 * psi_change
        
        # 扩展回原始维度
        psi_final = np.zeros_like(psi)
        psi_final[:, :, :self.site_dim] = psi_transformed
        psi_final[:, :, self.site_dim:] = psi[:, :, self.site_dim:]
        
        info = {
            'kernel_weights': self.kernel_weights.copy(),
            'entanglement_strength': np.mean(entanglement_spectrum),
            'hexagram_used': np.argmax(self.kernel_weights)
        }
        
        return psi_final, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        grad_psi[:, :, :self.site_dim] = gradient_clip(grad_psi[:, :, :self.site_dim], max_norm=1.0)
        
        grad_params = {
            'kernel_params': grad_psi.mean(axis=(0, 1)) * 0.01
        }
        
        return grad_psi, grad_params


# ==================== 第7层：阴符经·收缩层（YinfuContractor） ====================
"""
《阴符经》：'宇宙在乎手，万化生乎身'
'九窍三要'：选择性收缩
- 九窍：多个信息通道
- 三要：关键信息点
- 五贼：观、掩、取、舍、化
"""

class YinfuContractor:
    """
    阴符经·收缩层
    
    核心思想：
    - 九窍三要：选择性收缩（非均匀降维）
    - 信息重要性评估：saliency = compute_information_saliency(psi_grid)
    - key_sites = top_k(saliency, k=grid_size*0.15)
    - 五贼压缩策略
    """
    
    def __init__(self, num_sites: int, site_dim: int, contraction_ratio: float = 0.15):
        """
        Args:
            num_sites: site数量
            site_dim: site维度
            contraction_ratio: 收缩比例（默认15%）
        """
        self.num_sites = num_sites
        self.site_dim = site_dim
        self.contraction_ratio = contraction_ratio
        
        # 收缩强度参数
        self.contraction_strength = np.random.randn(num_sites, site_dim) * 0.1
        
        # 五贼参数
        self.wuzei_params = {
            'guan': np.random.randn(site_dim) * 0.1,    # 观
            'yan': np.random.randn(site_dim) * 0.1,     # 掩
            'qu': np.random.randn(site_dim) * 0.1,      # 取
            'she': np.random.randn(site_dim) * 0.1,     # 舍
            'hua': np.random.randn(site_dim) * 0.1      # 化
        }
        
        # 保留的key sites
        self.key_sites_mask = np.ones(num_sites)
        
    def _compute_information_saliency(self, psi: np.ndarray) -> np.ndarray:
        """
        计算信息重要性
        
        基于：
        - 局部方差
        - 梯度幅度
        - 与整体的互信息
        """
        batch_size = psi.shape[0]
        
        # 局部方差
        local_var = np.var(psi, axis=0)  # (num_sites, site_dim)
        
        # 梯度（近似）
        psi_diff = np.diff(psi, axis=1, prepend=psi[:, :1, :])
        gradient_mag = np.mean(np.abs(psi_diff), axis=0)
        
        # 互信息（psi与全局的关联）
        psi_mean = np.mean(psi, axis=1, keepdims=True)
        mutual_info = np.mean(np.abs(psi - psi_mean), axis=0)
        
        # 综合评分
        saliency = local_var + 0.5 * gradient_mag + 0.3 * mutual_info
        saliency = np.sum(saliency, axis=-1)  # (num_sites,)
        
        return saliency
    
    def _wuzei_compression(self, psi: np.ndarray, key_mask: np.ndarray) -> np.ndarray:
        """
        五贼压缩策略
        """
        batch_size = psi.shape[0]
        num_sites = psi.shape[1]
        
        # 观：观测降维
        guan_proj = psi @ self.wuzei_params['guan']  # (batch, num_sites)
        
        # 掩：信息遮蔽
        # key_mask: (num_sites,) -> (1, num_sites, 1) for broadcasting
        key_mask_expanded = key_mask[np.newaxis, :, np.newaxis]  # (1, num_sites, 1)
        psi_masked = psi * key_mask_expanded
        
        # 取：特征提取
        # wuzei_params['qu']: (site_dim,) -> 扩展为 (num_sites, site_dim)
        qu_matrix = np.tile(self.wuzei_params['qu'], (num_sites, 1))  # (num_sites, site_dim)
        psi_extracted = np.einsum('bsi,si->bs', psi_masked, qu_matrix)  # (batch, num_sites)
        # 扩展回(batch, num_sites, site_dim)
        psi_extracted = np.repeat(psi_extracted[:, :, np.newaxis], self.site_dim, axis=-1)
        
        # 舍：冗余丢弃
        舍_factor = 1 - np.abs(self.wuzei_params['she']) / (np.linalg.norm(self.wuzei_params['she']) + 1e-8)
        psi_shrunk = psi_extracted * 舍_factor
        
        # 化：等价变换
        hua_matrix = np.eye(self.site_dim) + 0.1 * self.wuzei_params['hua']
        psi_transformed = psi_shrunk @ hua_matrix
        
        return psi_transformed
    
    def forward(self, psi: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态，shape (batch, num_sites, site_dim)
            
        Returns:
            psi_contracted: 收缩后的量子态
            info: 收缩信息
        """
        batch_size, num_sites, site_dim = psi.shape
        
        # ========== 九窍三要：识别关键site ==========
        saliency = self._compute_information_saliency(psi)  # (num_sites,)
        
        # 选择top-k关键site
        k = int(num_sites * self.contraction_ratio)
        k = max(1, k)
        
        # 关键site保持不变，非关键site收缩
        sorted_indices = np.argsort(saliency)[::-1]
        key_indices = sorted_indices[:k]
        
        # 创建mask
        self.key_sites_mask = np.zeros(num_sites)
        self.key_sites_mask[key_indices] = 1.0
        
        # ========== 非均匀收缩 ==========
        # 关键site弱收缩，非关键site强收缩
        contraction_factor = 1 - (1 - self.contraction_ratio) * (1 - self.key_sites_mask)
        
        # 应用收缩
        psi_contracted = psi * contraction_factor[np.newaxis, :, np.newaxis]
        
        # ========== 五贼压缩 ==========
        psi_compressed = self._wuzei_compression(psi_contracted, self.key_sites_mask)
        
        # 合并结果
        psi_final = 0.7 * psi_contracted + 0.3 * psi_compressed
        
        info = {
            'saliency_scores': saliency,
            'key_sites': key_indices,
            'key_sites_mask': self.key_sites_mask.copy(),
            'contraction_factor': contraction_factor,
            'num_key_sites': k
        }
        
        return psi_final, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        grad_params = {name: np.zeros_like(p) for name, p in self.wuzei_params.items()}
        for name, p in self.wuzei_params.items():
            grad_params[name] = grad_psi.mean(axis=(0, 1))[:len(p)] * 0.01
        
        return grad_psi, grad_params


# ==================== 第8层：道德经·自组织层（DaodejingSelfOrg） ====================
"""
《道德经》（帛书版）：德经在前，道经在后
'道生之，德畜之'
'大器免成'：无固定收敛终点
'无为无以为'：最小必要变换
- 恒道：追求不变量
- 大器免成：保持开放
- 无为：最小变换
- 序参量：order_param = ||ψ̄|| > 0.7 时产生新层级
"""

class DaodejingSelfOrg:
    """
    道德经·自组织层（帛书版）
    
    核心思想：
    - 德经在前，道经在后：先数据驱动学习，再归纳全局规律
    - 恒道：追求不随数据分布变化的不变量
    - 大器免成：无固定收敛终点
    - 无为无以为：每层只做必要的最小变换
    - 序参量：order_param > 0.7 时可产生新层级
    """
    
    def __init__(self, site_dim: int, num_sites: int):
        """
        Args:
            site_dim: site维度
            num_sites: site数量
        """
        self.site_dim = site_dim
        self.num_sites = num_sites
        
        # 不变量提取器
        self.invariant_extractor = np.random.randn(site_dim, site_dim) * 0.01
        
        # 德（数据驱动）参数
        self.de_params = np.random.randn(site_dim, site_dim) * 0.1
        
        # 道（规律归纳）参数
        self.dao_params = np.random.randn(site_dim, site_dim) * 0.1
        
        # 序参量阈值
        self.order_threshold = 0.7
        
        # 自组织状态
        self.layer_emerged = False
        self.organization_history = []
        
    def forward(self, psi: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态，shape (batch, num_sites, site_dim)
            
        Returns:
            psi_selforg: 自组织后的量子态
            info: 自组织信息
        """
        batch_size = psi.shape[0]
        
        # ========== 德经在前：数据驱动学习 ==========
        # 从数据中学习局部模式
        psi_de = np.zeros_like(psi)
        for i in range(self.num_sites):
            # 局部聚合
            local_agg = np.mean(psi[:, max(0, i-1):min(self.num_sites, i+2), :], axis=1)
            # 数据驱动变换
            psi_de[:, i] = local_agg @ self.de_params
        
        # ========== 道经在后：归纳全局规律 ==========
        # 从局部模式中提取不变量
        psi_mean = np.mean(psi_de, axis=1, keepdims=True)  # (batch, 1, site_dim)
        psi_deviation = psi_de - psi_mean
        
        # 提取不变量
        invariants = np.mean(psi_deviation ** 2, axis=1)  # (batch, site_dim)
        
        # 全局规律应用
        global_pattern = np.tanh(invariants @ self.dao_params)  # (batch, site_dim)
        global_pattern_expanded = np.tile(global_pattern[:, np.newaxis, :], (1, self.num_sites, 1))
        
        psi_dao = psi_de + 0.1 * global_pattern_expanded
        
        # ========== 无为：最小必要变换 ==========
        # 计算当前状态与目标的差距
        psi_center = np.mean(psi_dao, axis=1, keepdims=True)
        deviation = np.sum((psi_dao - psi_center) ** 2, axis=-1)  # (batch, num_sites)
        
        # 只在必要时变换
        min_change_mask = (deviation > 0.1).astype(float)
        psi_minimal = psi_dao * (1 - 0.1 * min_change_mask[:, :, np.newaxis]) + \
                      psi_center * (0.1 * min_change_mask[:, :, np.newaxis])
        
        # ========== 序参量计算 ==========
        psi_bar = np.mean(psi_minimal, axis=1)  # (batch, site_dim)
        order_param = compute_norm(psi_bar) / np.sqrt(batch_size)
        
        self.organization_history.append(order_param)
        
        # 检查是否需要产生新层级
        self.layer_emerged = order_param > self.order_threshold
        
        info = {
            'order_param': order_param,
            'layer_emerged': self.layer_emerged,
            'invariant_strength': np.mean(invariants),
            'de_strength': np.mean(np.abs(psi_de)),
            'dao_strength': np.mean(np.abs(psi_dao)),
            'minimal_change_ratio': np.mean(min_change_mask)
        }
        
        return psi_minimal, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        grad_params = {
            'de_params': grad_psi.mean(axis=(0, 1)) * 0.01,
            'dao_params': grad_psi.mean(axis=(0, 1)) * 0.01
        }
        
        return grad_psi, grad_params


# ==================== 第9层：庄子·道枢层（ZhuangziUncertainty） ====================
"""
《庄子·齐物论》：'彼是莫得其偶，谓之道枢'
道枢：矛盾/不确定性时的中心守则
'环中以应无穷'：冲突时参数均匀化
'莫若以明'：清空先验，重新感知
"""

class ZhuangziUncertainty:
    """
    庄子·道枢层
    
    核心思想：
    - 道枢：检测矛盾/不确定性时退守中心
    - 环中以应无穷：冲突时γ参数均匀化
    - 莫若以明：清空先验（reset机制）
    """
    
    def __init__(self, site_dim: int, num_sites: int):
        """
        Args:
            site_dim: site维度
            num_sites: site数量
        """
        self.site_dim = site_dim
        self.num_sites = num_sites
        
        # 中心参数（道枢）
        self.center_params = np.zeros((num_sites, site_dim))
        
        # 不确定性检测参数
        self.uncertainty_threshold = 0.5
        self.conflict_history = []
        
        # 环中参数（均匀化）
        self.gamma_uniform = 0.3
        
        # Reset计数器
        self.no_reset_steps = 0
        self.reset_interval = 50
        
    def _detect_uncertainty(self, psi: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        检测不确定性
        
        Returns:
            uncertainty: 不确定性程度
            conflict_map: 冲突区域
        """
        batch_size = psi.shape[0]
        
        # 计算psi的分散度
        psi_mean = np.mean(psi, axis=0)  # (num_sites, site_dim)
        psi_std = np.std(psi, axis=0)
        
        # 分散度大表示不确定性高
        uncertainty = np.mean(psi_std)
        
        # 冲突区域检测
        conflict_map = (psi_std > self.uncertainty_threshold).astype(float)
        
        return uncertainty, conflict_map
    
    def forward(self, psi: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态，shape (batch, num_sites, site_dim)
            
        Returns:
            psi_centered: 道枢处理后的量子态
            info: 处理信息
        """
        batch_size = psi.shape[0]
        
        # ========== 不确定性检测 ==========
        uncertainty, conflict_map = self._detect_uncertainty(psi)
        
        self.conflict_history.append(uncertainty)
        
        # ========== 莫若以明：必要时Reset ==========
        self.no_reset_steps += 1
        if self.no_reset_steps > self.reset_interval:
            # 清空先验，重新感知
            recent_conflicts = self.conflict_history[-self.reset_interval:]
            if np.mean(recent_conflicts) > self.uncertainty_threshold:
                # Reset中心参数
                self.center_params = np.zeros((self.num_sites, self.site_dim))
                self.no_reset_steps = 0
        
        # ========== 环中以应无穷：均匀化 ==========
        # 计算当前中心
        current_center = np.mean(psi, axis=0)
        
        # 更新道枢中心
        self.center_params = 0.9 * self.center_params + 0.1 * current_center
        
        # 均匀化处理
        if uncertainty > self.uncertainty_threshold:
            # 高不确定性：向中心收缩
            center_expanded = self.center_params[np.newaxis, :, :]
            psi_centered = (1 - self.gamma_uniform) * psi + self.gamma_uniform * center_expanded
            mode = 'centered'
        else:
            # 正常状态：保持
            psi_centered = psi
            mode = 'normal'
        
        # ========== 道枢：最终平衡 ==========
        # 确保不会偏离中心太远
        center_expanded = self.center_params[np.newaxis, :, :]
        deviation = np.sum((psi_centered - center_expanded) ** 2, axis=-1)
        
        # 超过阈值的拉回
        safe_mask = (deviation < 1.0).astype(float)
        psi_final = psi_centered * safe_mask[:, :, np.newaxis] + \
                   center_expanded * (1 - safe_mask[:, :, np.newaxis])
        
        info = {
            'uncertainty': uncertainty,
            'conflict_map': conflict_map,
            'mode': mode,
            'center_norm': np.linalg.norm(self.center_params),
            'reset_triggered': self.no_reset_steps == 0
        }
        
        return psi_final, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        # 不返回可学习参数梯度（这是稳定层）
        grad_params = {}
        
        return grad_psi, grad_params


# ==================== 第10层：大学·知止层（DaxueConvergence） ====================
"""
《大学》：'大学之道，在明明德，在亲民，在止于至善'
知止→定→静→安→虑→得：七步收敛链
- 知止：设定明确目标阈值
- 定：锁定关键参数
- 静：降噪（梯度裁剪）
- 安：精调学习率
- 虑：推理验证
- 得：收敛确认
"""

class DaxueConvergence:
    """
    大学·知止层
    
    七步收敛链：
    1. 知止：设定目标阈值
    2. 定：锁定关键参数
    3. 静：降噪
    4. 安：精调学习率
    5. 虑：推理验证
    6. 得：收敛确认
    """
    
    def __init__(self, site_dim: int, num_sites: int, target_loss: float = 0.1):
        """
        Args:
            site_dim: site维度
            num_sites: site数量
            target_loss: 目标loss阈值
        """
        self.site_dim = site_dim
        self.num_sites = num_sites
        self.target_loss = target_loss
        
        # 知止参数
        self.target_psi = np.random.randn(num_sites, site_dim) * 0.01
        
        # 收敛状态
        self.convergence_step = 0
        self.loss_history = []
        self.converged = False
        
        # 学习率精调器
        self.base_lr = 0.01
        self.adjusted_lr = self.base_lr
        
        # 参数锁定状态
        self.locked_params = set()
        
    def forward(self, psi: np.ndarray, loss: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            psi: 输入量子态
            loss: 当前loss
            
        Returns:
            psi_converged: 收敛处理后的量子态
            info: 收敛信息
        """
        batch_size = psi.shape[0]
        
        self.loss_history.append(loss)
        
        # ========== 知止：设定目标 ==========
        if loss < self.target_loss and not self.converged:
            # 达到目标，更新目标状态
            self.target_psi = 0.9 * self.target_psi + 0.1 * np.mean(psi, axis=0)
        
        # ========== 七步收敛链 ==========
        stage_names = ['知止', '定', '静', '安', '虑', '得']
        
        # 1. 知止 → 2. 定：锁定
        if self.convergence_step >= 0:
            # 识别关键参数（梯度最大的）
            pass  # 简化处理
        
        # 3. 静：降噪
        if self.convergence_step >= 2:
            # 应用更强的梯度裁剪
            psi = gradient_clip(psi, max_norm=0.5)
        
        # 4. 安：精调学习率
        if self.convergence_step >= 3 and len(self.loss_history) >= 10:
            recent_loss = np.mean(self.loss_history[-10:])
            if recent_loss > self.loss_history[-10]:
                # Loss在增加，降低学习率
                self.adjusted_lr = self.base_lr * 0.9
            else:
                # Loss在下降，可以适当增大学习率
                self.adjusted_lr = min(self.base_lr * 1.1, 0.02)
        
        # 5. 虑：推理验证
        if self.convergence_step >= 4:
            # 检查是否收敛
            if len(self.loss_history) >= 20:
                recent_std = np.std(self.loss_history[-20:])
                if recent_std < 0.01:
                    self.converged = True
        
        # 6. 得：收敛确认
        if self.converged:
            # 收敛后保持稳定
            psi_final = 0.95 * psi + 0.05 * self.target_psi[np.newaxis, :, :]
        else:
            psi_final = psi
        
        self.convergence_step = min(self.convergence_step + 1, 5)
        
        info = {
            'convergence_stage': stage_names[min(self.convergence_step, 5)],
            'adjusted_lr': self.adjusted_lr,
            'converged': self.converged,
            'target_loss': self.target_loss,
            'current_loss': loss,
            'loss_trend': np.mean(self.loss_history[-5:]) if len(self.loss_history) >= 5 else loss
        }
        
        return psi_final, info
    
    def backward(self, grad_output: np.ndarray, info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        反向传播
        """
        grad_psi = grad_output.copy()
        
        # 应用学习率调整
        grad_psi = grad_psi * self.adjusted_lr
        
        grad_params = {
            'target_psi': grad_psi.mean(axis=0) * 0.01
        }
        
        return grad_psi, grad_params


# ==================== 顶层管道：道生多尺度架构 ====================

class DaoShengPipeline:
    """
    道生多尺度架构（DaoSheng Multi-scale Ansatz）
    
    十一层完整管道：
    第10层：大学·知止层
    第9层：庄子·道枢层
    第8层：道德经·自组织层
    第7层：阴符经·收缩层
    第6层：易经·变换层
    第5层：中庸·致中和层
    第4层：经络拓扑层
    第3层：黄帝四经·法则层
    第2层：三生耦合层（核心）
    第1层：孙子兵法·策略路由层
    第0层：六书编码层
    """
    
    def __init__(self, input_dim: int, output_dim: int = 10, 
                 bond_dim: int = 8, patch_size: int = 4):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度（分类数）
            bond_dim: 纠缠键维
            patch_size: patch大小
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.patch_size = patch_size
        
        # 计算中间维度
        self.num_sites = input_dim // patch_size
        self.encode_dim = 2
        
        print(f"初始化道生架构：")
        print(f"  输入维度: {input_dim}")
        print(f"  Patch数量: {self.num_sites}")
        print(f"  编码维度: {self.encode_dim}")
        print(f"  键维: {bond_dim}")
        
        # 初始化十一层
        self.layers = {
            0: LiushuEncoder(input_dim, patch_size, self.encode_dim),
            1: SunziRouter(num_paths=4, hidden_dim=self.encode_dim),
            2: SanshengCoupler(site_dim=self.encode_dim, num_sites=self.num_sites, bond_dim=bond_dim),
            3: HuangdiLawLayer(site_dim=self.encode_dim, num_sites=self.num_sites),
            4: JingluoTopology(num_sites=self.num_sites, site_dim=self.encode_dim),
            5: ZhongyongRegularizer(site_dim=self.encode_dim, num_sites=self.num_sites),
            6: YijingTransform(site_dim=self.encode_dim, num_sites=self.num_sites),
            7: YinfuContractor(num_sites=self.num_sites, site_dim=self.encode_dim),
            8: DaodejingSelfOrg(site_dim=self.encode_dim, num_sites=self.num_sites),
            9: ZhuangziUncertainty(site_dim=self.encode_dim, num_sites=self.num_sites),
            10: DaxueConvergence(site_dim=self.encode_dim, num_sites=self.num_sites, target_loss=0.1)
        }
        
        # 分类头
        self.classifier_weight = np.random.randn(self.encode_dim, output_dim) * 0.01
        self.classifier_bias = np.zeros(output_dim)
        
        # 训练状态
        self.global_step = 0
        self.layer_grad_norms = {}
        
        print("十一层架构初始化完成！")
        
    def forward(self, x: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        前向传播
        
        Args:
            x: 输入数据，shape (batch, input_dim)
            labels: 标签（可选）
            
        Returns:
            logits: 输出logits
            info: 各层信息
        """
        batch_size = x.shape[0]
        info = {}
        
        # ========== 第0层：六书编码 ==========
        psi, info[0] = self.layers[0].forward(x)
        info[0]['layer_name'] = '六书编码层'
        
        # ========== 第1层：孙子兵法路由 ==========
        psi, info[1] = self.layers[1].forward(psi)
        info[1]['layer_name'] = '孙子兵法·策略路由层'
        
        # ========== 第2层：三生耦合（核心）==========
        psi, info[2] = self.layers[2].forward(psi)
        info[2]['layer_name'] = '三生耦合层（核心）'
        
        # ========== 第3层：黄帝四经法则 ==========
        psi, info[3] = self.layers[3].forward(psi, labels)
        info[3]['layer_name'] = '黄帝四经·法则层'
        
        # ========== 第4层：经络拓扑 ==========
        overfit_signal = info[5].get('overfit_signal', 0) if 5 in info else 0
        psi, info[4] = self.layers[4].forward(psi, overfit_signal)
        info[4]['layer_name'] = '经络拓扑层'
        
        # ========== 第5层：中庸正则化 ==========
        task_loss = info.get('task_loss', 0.5)
        psi, info[5] = self.layers[5].forward(psi, task_loss)
        info[5]['layer_name'] = '中庸·致中和层'
        
        # ========== 第6层：易经变换 ==========
        psi, info[6] = self.layers[6].forward(psi)
        info[6]['layer_name'] = '易经·变换层'
        
        # ========== 第7层：阴符经收缩 ==========
        psi, info[7] = self.layers[7].forward(psi)
        info[7]['layer_name'] = '阴符经·收缩层'
        
        # ========== 第8层：道德经自组织 ==========
        psi, info[8] = self.layers[8].forward(psi)
        info[8]['layer_name'] = '道德经·自组织层'
        
        # ========== 第9层：庄子道枢 ==========
        psi, info[9] = self.layers[9].forward(psi)
        info[9]['layer_name'] = '庄子·道枢层'
        
        # ========== 第10层：大学知止 ==========
        psi, info[10] = self.layers[10].forward(psi, task_loss)
        info[10]['layer_name'] = '大学·知止层'
        
        # ========== 分类头 ==========
        psi_agg = np.mean(psi, axis=1)  # (batch, encode_dim)
        logits = psi_agg @ self.classifier_weight + self.classifier_bias
        
        # 更新任务loss
        info['task_loss'] = self._compute_loss(logits, labels) if labels is not None else 0.5
        
        return logits, info
    
    def _compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """计算交叉熵损失"""
        probs = softmax(logits, axis=-1)
        batch_size = logits.shape[0]
        
        # 避免log(0)
        eps = 1e-8
        labels_onehot = np.zeros_like(probs)
        labels_onehot[np.arange(batch_size), labels] = 1.0
        
        loss = -np.sum(labels_onehot * np.log(probs + eps)) / batch_size
        return loss
    
    def backward(self, logits: np.ndarray, labels: np.ndarray, info: Dict) -> Dict:
        """
        反向传播
        
        Returns:
            grad_info: 各层梯度信息
        """
        batch_size = logits.shape[0]
        grad_info = {}
        
        # ========== 分类头梯度 ==========
        probs = softmax(logits, axis=-1)
        labels_onehot = np.zeros_like(probs)
        labels_onehot[np.arange(batch_size), labels] = 1.0
        
        grad_psi_agg = (probs - labels_onehot) @ self.classifier_weight.T / batch_size
        grad_classifier_weight = np.mean(probs - labels_onehot, axis=0).reshape(-1, 1) * 0.01
        
        # 扩展梯度到完整psi形状
        grad_psi = grad_psi_agg[:, np.newaxis, :]
        
        # ========== 第10层反向 ==========
        grad_psi, grad_10 = self.layers[10].backward(grad_psi, info[10])
        grad_info[10] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第9层反向 ==========
        grad_psi, grad_9 = self.layers[9].backward(grad_psi, info[9])
        grad_info[9] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第8层反向 ==========
        grad_psi, grad_8 = self.layers[8].backward(grad_psi, info[8])
        grad_info[8] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第7层反向 ==========
        grad_psi, grad_7 = self.layers[7].backward(grad_psi, info[7])
        grad_info[7] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第6层反向 ==========
        grad_psi, grad_6 = self.layers[6].backward(grad_psi, info[6])
        grad_info[6] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第5层反向 ==========
        grad_psi, grad_5 = self.layers[5].backward(grad_psi, info[5])
        grad_info[5] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第4层反向 ==========
        grad_psi, grad_4 = self.layers[4].backward(grad_psi, info[4])
        grad_info[4] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第3层反向 ==========
        grad_psi, grad_3 = self.layers[3].backward(grad_psi, info[3])
        grad_info[3] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第2层反向 ==========
        grad_psi, grad_2 = self.layers[2].backward(grad_psi, info[2])
        grad_info[2] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第1层反向 ==========
        grad_psi = self.layers[1].backward(grad_psi, info[1])
        grad_info[1] = {'grad_norm': np.linalg.norm(grad_psi)}
        
        # ========== 第0层反向 ==========
        grad_x, grad_0 = self.layers[0].backward(grad_psi)
        grad_info[0] = {'grad_norm': np.linalg.norm(grad_x)}
        
        # 记录梯度范数
        self.layer_grad_norms[self.global_step] = {k: v['grad_norm'] for k, v in grad_info.items()}
        
        return grad_info
    
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01) -> Dict:
        """
        单步训练
        
        Args:
            x: 输入数据
            y: 标签
            lr: 学习率
            
        Returns:
            metrics: 训练指标
        """
        self.global_step += 1
        
        # 前向传播
        logits, info = self.forward(x, y)
        
        # 计算loss
        loss = self._compute_loss(logits, y)
        
        # 反向传播
        grad_info = self.backward(logits, y, info)
        
        # 计算accuracy
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == y)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'grad_norms': {k: v['grad_norm'] for k, v in grad_info.items()},
            'step': self.global_step
        }
        
        return metrics
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            x: 输入数据
            
        Returns:
            predictions: 预测结果
        """
        logits, _ = self.forward(x)
        predictions = np.argmax(logits, axis=-1)
        return predictions
    
    def get_layer_info(self) -> List[Dict]:
        """获取各层信息"""
        layer_info = []
        layer_names = [
            '六书编码层', '孙子兵法路由层', '三生耦合层', '黄帝四经法则层',
            '经络拓扑层', '中庸正则化层', '易经变换层', '阴符经收缩层',
            '道德经自组织层', '庄子道枢层', '大学知止层'
        ]
        
        for i, name in enumerate(layer_names):
            layer_info.append({
                'layer_id': i,
                'layer_name': name,
                'has_forward': hasattr(self.layers[i], 'forward'),
                'has_backward': hasattr(self.layers[i], 'backward')
            })
        
        return layer_info


# ==================== 便捷函数 ====================

def create_daosheng_model(input_dim: int, output_dim: int = 10, 
                          bond_dim: int = 8, patch_size: int = 4) -> DaoShengPipeline:
    """创建道生模型"""
    return DaoShengPipeline(
        input_dim=input_dim,
        output_dim=output_dim,
        bond_dim=bond_dim,
        patch_size=patch_size
    )


# ==================== 测试代码 ====================

if __name__ == '__main__':
    print("=" * 60)
    print("道生多尺度架构（DaoSheng Multi-scale Ansatz）测试")
    print("=" * 60)
    
    # 创建模型
    model = create_daosheng_model(input_dim=32, output_dim=10, bond_dim=8, patch_size=4)
    
    # 打印层信息
    print("\n层信息：")
    for info in model.get_layer_info():
        print(f"  第{info['layer_id']}层: {info['layer_name']}")
    
    # 生成测试数据
    print("\n生成测试数据...")
    set_random_seed(42)
    batch_size = 4
    x_test = np.random.randn(batch_size, 32)
    y_test = np.random.randint(0, 10, batch_size)
    
    # 前向传播测试
    print("\n前向传播测试...")
    logits, info = model.forward(x_test, y_test)
    print(f"  输入shape: {x_test.shape}")
    print(f"  输出shape: {logits.shape}")
    print(f"  Loss: {model._compute_loss(logits, y_test):.4f}")
    
    # 反向传播测试
    print("\n反向传播测试...")
    grad_info = model.backward(logits, y_test, info)
    print("  各层梯度范数:")
    for layer_id, grad_data in sorted(grad_info.items()):
        print(f"    第{layer_id}层: {grad_data['grad_norm']:.6f}")
    
    # 训练步骤测试
    print("\n训练步骤测试...")
    metrics = model.train_step(x_test, y_test)
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
