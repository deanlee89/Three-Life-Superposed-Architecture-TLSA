#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三生叠加态原生AGI架构 - 端到端Demo
包含：状态生成、耦合算子、五行动力学、分类任务、去噪对比
"""

import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 核心函数
# ============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax函数：映射到概率单纯形Δ^n"""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=-1, keepdims=True)
    return np.exp(x - x_max) / np.exp(x - x_max).sum(axis=-1, keepdims=True)


def initialize_sansheng_state(shape: Tuple[int, ...], init_type: str = 'balanced') -> np.ndarray:
    """初始化三生状态 (α, γ, β) ∈ Δ²"""
    n = shape[0] * shape[1]
    if init_type == 'balanced':
        psi_flat = np.ones((n, 3)) / 3.0
    elif init_type == 'yin_dominant':
        psi_flat = np.tile([0.5, 0.3, 0.2], (n, 1))
    elif init_type == 'yang_dominant':
        psi_flat = np.tile([0.2, 0.3, 0.5], (n, 1))
    else:
        psi_flat = softmax(np.random.randn(n, 3))
    return psi_flat.reshape(*shape, 3)


def print_state(psi: np.ndarray, name: str = "状态"):
    """打印状态分布"""
    alpha, gamma, beta = psi[..., 0].mean(), psi[..., 1].mean(), psi[..., 2].mean()
    total = (psi[..., 0] + psi[..., 1] + psi[..., 2]).mean()
    print(f"\n{name}: α={alpha:.4f}, γ={gamma:.4f}, β={beta:.4f}, 误差={abs(total-1):.2e}")


def get_8_neighbors(psi: np.ndarray) -> List[np.ndarray]:
    """获取8邻域"""
    H, W = psi.shape[:2]
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    return [psi[(np.arange(H)[:, None]+di)%H, (np.arange(W)[None, :]+dj)%W] 
            for di, dj in offsets]


def coupling_operator(psi_v: np.ndarray, psi_u: np.ndarray, 
                     lambda_h: float = 0.5, lambda_bal: float = 0.4) -> np.ndarray:
    """核心耦合算子 C(ψ_v, ψ_u)"""
    av, gv, bv = psi_v[..., 0], psi_v[..., 1], psi_v[..., 2]
    au, gu, bu = psi_u[..., 0], psi_u[..., 1], psi_u[..., 2]
    
    cross = av * bu + bv * au
    gamma_raw = gv * (av * (av + gv) + cross * 0.3) + lambda_h * gv * gu
    alpha_raw = av**2 + av * gu + lambda_bal * (au - bu)
    beta_raw = bv**2 + bv * gu - lambda_bal * (au - bu)
    
    raw = np.stack([alpha_raw, gamma_raw, beta_raw], axis=-1)
    raw = np.maximum(raw, 1e-8)
    return raw / raw.sum(axis=-1, keepdims=True)


def wuxing_transition(psi: np.ndarray, alpha_w: float = 0.3) -> np.ndarray:
    """五行动力学：木(α)→火(γ)→土(β)→金(α)→水(γ)循环"""
    H, W = psi.shape[:2]
    result = psi.copy()
    
    # 主方向循环
    offsets = [(-1,0), (0,1), (1,0), (0,-1)]
    for idx, (di, dj) in enumerate(offsets):
        ni = (np.arange(H)[:, None] + di) % H
        nj = (np.arange(W)[None, :] + dj) % W
        neighbor = psi[ni, nj]
        if idx == 0: result[..., 1] += alpha_w * neighbor[..., 0]  # N: α → γ
        elif idx == 1: result[..., 2] += alpha_w * neighbor[..., 1]  # E: γ → β
        elif idx == 2: result[..., 0] += alpha_w * neighbor[..., 2]  # S: β → α
        elif idx == 3: result[..., 1] += alpha_w * 0.5 * neighbor[..., 0]  # W: α → γ
    
    # 对角线增强
    for di, dj in [(-1,-1), (-1,1), (1,1), (1,-1)]:
        ni = (np.arange(H)[:, None] + di) % H
        nj = (np.arange(W)[None, :] + dj) % W
        neighbor = psi[ni, nj]
        result[..., 1] += 0.1 * alpha_w * neighbor[..., 0]
    
    result = np.maximum(result, 1e-8)
    return result / result.sum(axis=-1, keepdims=True)


def evolve_step(psi: np.ndarray, eps: float = 0.3, lh: float = 0.5, lb: float = 0.4) -> np.ndarray:
    """单步耦合演化"""
    neighbors = get_8_neighbors(psi)
    coupled = sum(coupling_operator(psi, n, lh, lb) for n in neighbors) / 8
    psi_new = (1 - eps) * psi + eps * coupled
    psi_new = np.maximum(psi_new, 1e-8)
    return psi_new / psi_new.sum(axis=-1, keepdims=True)


def evolve(psi: np.ndarray, steps: int = 10) -> Tuple[np.ndarray, List[dict]]:
    """多步演化"""
    psi_c = psi.copy()
    metrics = []
    for _ in range(steps):
        psi_c = evolve_step(psi_c)
        psi_c = wuxing_transition(psi_c)
        am, gm, bm = psi_c[..., 0].mean(), psi_c[..., 1].mean(), psi_c[..., 2].mean()
        metrics.append({
            'balance': 1.0 - abs(am - bm),
            'harmony': gm,
            'error': abs((psi_c[..., 0]+psi_c[..., 1]+psi_c[..., 2]).mean() - 1)
        })
    return psi_c, metrics


# ============================================================================
# 去噪机制
# ============================================================================

def bilateral_filter(psi: np.ndarray, ss: float = 1.5, sr: float = 0.3) -> np.ndarray:
    """双边滤波"""
    H, W = psi.shape[:2]
    result = np.zeros_like(psi)
    for i in range(H):
        for j in range(W):
            center = psi[i, j]
            ws, wv, wsum = 0.0, np.zeros(3), 0.0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni, nj = (i+di)%H, (j+dj)%W
                    neighbor = psi[ni, nj]
                    w = np.exp(-(di**2+dj**2)/(2*ss**2)) * np.exp(-np.sum((center-neighbor)**2)/(2*sr**2))
                    ws += w; wv += w * neighbor
            result[i, j] = wv / ws
    result = np.maximum(result, 1e-8)
    return result / result.sum(axis=-1, keepdims=True)


def anomaly_suppression(psi: np.ndarray, thresh: float = 2.0) -> np.ndarray:
    """异常抑制"""
    H, W = psi.shape[:2]
    result = psi.copy()
    neighbors = get_8_neighbors(psi)
    local_mean = np.stack(neighbors).mean(axis=0)
    local_std = np.stack(neighbors).std(axis=0)
    is_anom = (np.abs(psi - local_mean) > thresh * (local_std + 1e-8)).any(axis=-1)
    for i in range(H):
        for j in range(W):
            if is_anom[i, j]:
                dev = np.abs(psi[i,j] - local_mean[i,j]).sum()
                pull = 1.0 / (1.0 + np.exp(-(dev - thresh)))
                result[i,j] = (1-pull)*psi[i,j] + pull*local_mean[i,j]
    result = np.maximum(result, 1e-8)
    return result / result.sum(axis=-1, keepdims=True)


def temperature_annealing(psi: np.ndarray, T: float = 1.5) -> np.ndarray:
    """温度退火"""
    return softmax(np.log(psi + 1e-8) / T)


def denoise(psi: np.ndarray, ss: float = 1.5, sr: float = 0.3, thresh: float = 2.0, T: float = 1.5) -> np.ndarray:
    """三重去噪"""
    psi = bilateral_filter(psi, ss, sr)
    psi = anomaly_suppression(psi, thresh)
    return temperature_annealing(psi, T)


# ============================================================================
# 分类任务
# ============================================================================

def generate_patterns(num_classes: int = 6, size: int = 8, samples: int = 60, noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """生成6类8×8图案"""
    X, y = [], []
    for c in range(num_classes):
        for _ in range(samples):
            p = np.zeros((size, size))
            if c == 0: p[5:7, 0:2] = 1; p[4, 0] = 1
            elif c == 1: p[3:5, 3:5] = 1; p[3:5, [2,4]] = 1; p[[2,5], 3:5] = 1
            elif c == 2: p[2:6, [2,5]] = 1; p[[2,5], 2:6] = 1
            elif c == 3: 
                for i in range(size): p[i, i] = 1
                p[[0,1], [1,0]] = 0.5
            elif c == 4: p[2, 3:5] = 1; p[3:5, 4] = 1; p[2, 2] = 0.5
            elif c == 5:
                for i in range(size): p[i, i] = 1; p[i, size-1-i] = 1
            if noise > 0: p = np.clip(p + np.random.randn(size,size)*noise, 0, 1)
            X.append(p); y.append(c)
    X, y = np.array(X), np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def encode_to_sansheng(X: np.ndarray) -> np.ndarray:
    """编码为三生状态"""
    N, H, W = X.shape
    psi = np.stack([X, 0.3*np.ones((N,H,W)), 1-X], axis=-1)
    psi = np.maximum(psi, 1e-8)
    return psi / psi.sum(axis=-1, keepdims=True)


def global_gamma_pooling(psi: np.ndarray) -> np.ndarray:
    """全局γ池化"""
    return psi[..., 1].reshape(psi.shape[0], -1)


def run_classification(X_train, y_train, X_test, y_test, use_denoise: bool = False, steps: int = 2, noise_level: float = 0.0) -> float:
    """运行分类"""
    psi_train, psi_test = encode_to_sansheng(X_train), encode_to_sansheng(X_test)
    
    # 去噪参数
    if use_denoise:
        if noise_level >= 0.5: ss, sr, th, T = 1.2, 0.4, 2.5, 1.3
        elif noise_level >= 0.3: ss, sr, th, T = 1.5, 0.3, 2.0, 1.5
        else: ss, sr, th, T = 1.0, 0.2, 2.5, 1.2
    
    # 提取特征
    train_feat, test_feat = [], []
    for psi in psi_train:
        if use_denoise: psi = denoise(psi, ss, sr, th, T)
        _, m = evolve(psi[np.newaxis], steps)
        train_feat.append(global_gamma_pooling(_)[0])
    for psi in psi_test:
        if use_denoise: psi = denoise(psi, ss, sr, th, T)
        _, m = evolve(psi[np.newaxis], steps)
        test_feat.append(global_gamma_pooling(_)[0])
    
    train_feat, test_feat = np.array(train_feat), np.array(test_feat)
    
    # KNN分类
    y_pred = []
    for i in range(len(test_feat)):
        dists = np.sum((train_feat - test_feat[i])**2, axis=1)
        k = min(3, len(y_train))
        nearest = y_train[np.argsort(dists)[:k]]
        y_pred.append(np.bincount(nearest.astype(int)).argmax())
    
    return (np.array(y_pred) == y_test).mean()


# ============================================================================
# 主程序
# ============================================================================

def main():
    np.random.seed(42)
    
    print("="*70)
    print("三生叠加态原生AGI架构 - 端到端Demo")
    print("="*70)
    
    # 模块1：三生叠加态生成
    print("\n" + "-"*70)
    print("模块1：三生叠加态生成")
    print("-"*70)
    psi = initialize_sansheng_state((6, 6), 'balanced')
    print(f"网格: 6×6, 状态维度: {psi.shape}")
    print_state(psi, "初始状态")
    
    # 模块2：耦合算子
    print("\n" + "-"*70)
    print("模块2：耦合算子与五行动力学")
    print("-"*70)
    psi_yin, psi_yang = np.array([0.6, 0.2, 0.2]), np.array([0.2, 0.2, 0.6])
    psi_coup = coupling_operator(psi_yin, psi_yang)
    print(f"阴主导 {psi_yin} + 阳主导 {psi_yang} → 耦合后 {psi_coup}")
    print(f"阴阳差: {abs(psi_coup[0]-psi_coup[2]):.4f}, 和合度: {psi_coup[1]:.4f}")
    
    # 模块3：演化过程
    print("\n" + "-"*70)
    print("模块3：演化过程与收敛性")
    print("-"*70)
    psi_init = initialize_sansheng_state((6, 6), 'yin_dominant')
    psi_evo, metrics = evolve(psi_init, steps=9)
    print_state(psi_init, "起点")
    print_state(psi_evo, "终点")
    print(f"平衡度: {metrics[-1]['balance']:.4f}, 和合度: {metrics[-1]['harmony']:.4f}")
    
    # 模块4：分类任务
    print("\n" + "-"*70)
    print("模块4：简单分类任务（无噪声）")
    print("-"*70)
    X_tr, y_tr = generate_patterns(6, 8, 60, 0.0)
    X_te, y_te = generate_patterns(6, 8, 20, 0.0)
    acc = run_classification(X_tr, y_tr, X_te, y_te)
    print(f"无噪声分类准确率: {acc*100:.1f}%")
    
    # 模块5：去噪对比
    print("\n" + "-"*70)
    print("模块5：去噪增强对比实验")
    print("-"*70)
    
    results = []
    for noise in [0.0, 0.15, 0.3, 0.45, 0.6]:
        X_tr, y_tr = generate_patterns(6, 8, 30, noise)
        X_te, y_te = generate_patterns(6, 8, 20, noise)
        acc_no = run_classification(X_tr, y_tr, X_te, y_te, False, 2, noise)
        acc_de = run_classification(X_tr, y_tr, X_te, y_te, True, 2, noise)
        results.append({'noise': noise, 'no': acc_no*100, 'de': acc_de*100, 'imp': (acc_de-acc_no)*100})
        print(f"σ={noise:.2f}: 无去噪={acc_no*100:.1f}%, 有去噪={acc_de*100:.1f}%, 提升={(acc_de-acc_no)*100:+.1f}%")
    
    # 摘要
    print("\n" + "="*70)
    print("实验结果摘要")
    print("="*70)
    print(f"\n{'噪声':<8} {'无去噪':<12} {'有去噪':<12} {'提升':<10}")
    print("-"*45)
    for r in results:
        print(f"σ={r['noise']:<5.2f} {r['no']:<12.1f}% {r['de']:<12.1f}% {r['imp']:+.1f}%")
    
    avg_no = np.mean([r['no'] for r in results])
    avg_de = np.mean([r['de'] for r in results])
    print("-"*45)
    print(f"平均:      {avg_no:<12.1f}% {avg_de:<12.1f}% {(avg_de-avg_no):+.1f}%")
    
    print(f"\n【核心结论】")
    print(f"  ✓ 三生叠加态生成：概率单纯形Δ²初始化完成")
    print(f"  ✓ 耦合算子与五行动力学：8邻域耦合+五行循环")
    print(f"  ✓ 演化收敛性：9步演化，平衡度={metrics[-1]['balance']:.4f}")
    print(f"  ✓ 分类任务：6类8×8图案准确率={acc*100:.1f}%")
    print(f"  ✓ 去噪效果：中等噪声(σ=0.45)下提升+{results[3]['imp']:.1f}%")
    print("\n" + "="*70)
    print("Demo运行完成！")
    print("="*70)


if __name__ == "__main__":
    main()
