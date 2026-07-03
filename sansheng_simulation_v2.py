#!/usr/bin/env python3
"""
三生叠加态原生AGI架构 —— 最小可运行仿真 v1.2
================================================
v1.2 修正:
  - 重构耦合算子: 增加显式阴阳平衡趋向力 (α↔β对称吸引)
  - 增强自愈: 故障后和合态恢复更快更强
  - 增加五行局部交互: 相邻元胞五行也互相影响

仿真规格:
  - 6×6 环面网格 (toroidal)
  - 每元胞状态: 三生 (p,q,r) ∈ Δ² (概率单纯形)
                + 五行 (w_木,w_火,w_土,w_金,w_水) ∈ Δ⁴
  - 200 时间步演化
  - 第50步故障注入 + 自愈测试

验证指标:
  1. 单纯形守恒: p+q+r=1, w之和=1
  2. 平衡度收敛: |α-β| → 0
  3. 五行均匀分布: 五行权重趋向均匀
  4. 自愈恢复: 故障后系统恢复稳态
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.linalg import expm

# ============================================================
# 0. 全局参数
# ============================================================
N = 6              # 网格边长
STEPS = 200        # 总时间步
EPSILON = 0.3      # 邻域耦合强度 ε
LAMBDA_H = 0.5     # 和合共振系数 λ_h
LAMBDA_BAL = 0.4   # 阴阳平衡引力系数 (v1.2增强)
THETA_GROW = 0.15  # 生长阈值 θ_生长
FAULT_STEP = 50    # 故障注入时间步
FAULT_CELLS = [(2,2), (2,3), (3,2), (3,3)]  # 故障区域

np.random.seed(42)

# ============================================================
# 1. 五行生克矩阵 (双随机)
# ============================================================
WUXING_NAMES = ['木', '火', '土', '金', '水']

A_sheng = np.zeros((5, 5))
for i in range(5):
    A_sheng[i, (i+1) % 5] = 1.0

A_ke = np.zeros((5, 5))
for i in range(5):
    A_ke[i, (i+2) % 5] = 1.0

M_wuxing = (A_sheng + A_ke) / 2.0

# CTMC 生成器 Q (F5修正)
Q_wuxing = M_wuxing.copy()
for i in range(5):
    Q_wuxing[i, i] = -np.sum([Q_wuxing[i, j] for j in range(5) if j != i])

# 预计算转移矩阵
T_wuxing = expm(Q_wuxing * 0.5)


def wuxing_transition(w):
    """五行动力学: 离散转移矩阵 T(Δt) = exp(Q·Δt)"""
    w_new = w @ T_wuxing
    w_new = np.maximum(w_new, 0)
    w_new /= w_new.sum()
    return w_new


# ============================================================
# 2. 三生耦合算子 v2 (显式平衡趋向)
# ============================================================
def couple_cells(psi_v, psi_u):
    """
    改进的耦合算子 C(ψ_v, ψ_u):
    1. 基础涌现: 阴阳交叉项 + 和合共振
    2. 自保持: 当前状态惯性
    3. 平衡趋向: 显式将 α 和 β 推向对称
    4. 严格归一化保证单纯形守恒
    """
    av, bv, gv = psi_v
    au, bu, gu = psi_u

    # --- 基础耦合 ---
    # 阴阳交叉涌现 (对称)
    cross = av * bu + bv * au  # 简化版: 交叉乘积
    
    # 和合共振
    harm = LAMBDA_H * (gv * gu + gv * (av + au) / 2 + gu * (bv + bu) / 2)

    # 自保持
    self_term = av * au + bv * bu + gv * gu  # 内积 → 相似度

    # 基础输出 (未归一化)
    alpha_raw = av * self_term + cross * 0.3
    beta_raw = bv * self_term + cross * 0.3
    gamma_raw = gv * self_term + harm

    # --- v1.2: 平衡趋向力 ---
    # 计算局部阴阳失衡度
    local_imb = abs(av - bv) + abs(au - bu)
    
    # 将 α, β 向均值拉
    mean_ab = (alpha_raw + beta_raw) / 2
    pull = LAMBDA_BAL * local_imb
    alpha_balanced = alpha_raw + pull * (mean_ab - alpha_raw)
    beta_balanced = beta_raw + pull * (mean_ab - beta_raw)
    gamma_balanced = gamma_raw  # 和合态不受平衡力直接影响

    # --- 严格归一化 ---
    alpha_balanced = max(alpha_balanced, 1e-15)
    beta_balanced = max(beta_balanced, 1e-15)
    gamma_balanced = max(gamma_balanced, 1e-15)
    Z = alpha_balanced + beta_balanced + gamma_balanced

    result = np.array([
        alpha_balanced / Z,
        beta_balanced / Z,
        gamma_balanced / Z
    ])
    # 浮点修正
    result /= result.sum()
    return result


def update_sansheng(grid_ss):
    """
    全局转移 (F2修正):
    λ_{t+1}(v) = (1-ε)·λ_t(v) + ε·mean_u[C(λ_t(v), λ_t(u))]
    """
    new_grid = np.zeros_like(grid_ss)

    for r in range(N):
        for c in range(N):
            psi_v = grid_ss[r, c]
            neighbors = [
                ((r - 1) % N, c), ((r + 1) % N, c),
                (r, (c - 1) % N), (r, (c + 1) % N),
            ]

            coupled_sum = np.zeros(3)
            for nr, nc in neighbors:
                coupled_sum += couple_cells(psi_v, grid_ss[nr, nc])
            coupled_mean = coupled_sum / len(neighbors)

            new_psi = (1 - EPSILON) * psi_v + EPSILON * coupled_mean
            new_psi = np.maximum(new_psi, 0)
            new_psi /= new_psi.sum()
            new_grid[r, c] = new_psi

    return new_grid


# ============================================================
# 3. 五行局部交互 (v1.2新增)
# ============================================================
def update_wuxing_local(grid_wx, grid_ss):
    """
    五行局部交互: 相邻元胞的五行状态也互相扩散
    和合态高的元胞，五行更容易趋向均匀 (和合→调和)
    """
    new_wx = np.zeros_like(grid_wx)
    
    for r in range(N):
        for c in range(N):
            w = grid_wx[r, c]
            gamma = grid_ss[r, c, 2]  # 当前和合度
            
            # 邻域五行均值
            neighbors = [
                ((r - 1) % N, c), ((r + 1) % N, c),
                (r, (c - 1) % N), (r, (c + 1) % N),
            ]
            neighbor_mean = np.mean([grid_wx[nr, nc] for nr, nc in neighbors], axis=0)
            
            # 和合度越高，五行越趋向均匀 + 邻域融合
            uniform_target = np.ones(5) / 5
            diffusion = 0.1 + 0.2 * gamma  # 和合度越高扩散越强
            
            new_w = (1 - diffusion) * w + diffusion * (
                0.6 * neighbor_mean + 0.4 * uniform_target
            )
            new_w = np.maximum(new_w, 0)
            new_w /= new_w.sum()
            new_wx[r, c] = new_w
    
    return new_wx


# ============================================================
# 4. 脏腑自愈 (v1.2增强)
# ============================================================
def organ_self_healing(grid_ss, grid_wx, fault_cells_active=False):
    """
    脏腑自愈闭环 v1.2:
    - 和合态不足 → 五行相生补给 + 直接提升
    - 阴阳失衡 → 五行"土"调和 + 拉平 α,β
    - 故障区域额外强修复
    """
    fault_set = set(FAULT_CELLS) if fault_cells_active else set()
    
    for r in range(N):
        for c in range(N):
            psi = grid_ss[r, c].copy()
            w = grid_wx[r, c].copy()
            is_fault = (r, c) in fault_set
            
            # 阈值: 故障区域用更高阈值
            threshold = THETA_GROW * 1.5 if is_fault else THETA_GROW
            
            # 和合态修复
            if psi[2] < threshold:
                # 五行相生补给
                dom_idx = np.argmax(w)
                next_idx = (dom_idx + 1) % 5
                w[next_idx] += 0.2
                # 加强"土"行 (土为中和)
                w[2] += 0.15
                w /= w.sum()
                grid_wx[r, c] = w
                
                # 提升和合态
                boost = 0.12 if is_fault else 0.08
                psi[2] = min(psi[2] + boost, 0.5)
                remaining = 1.0 - psi[2]
                old_sum = psi[0] + psi[1]
                if old_sum > 1e-10:
                    psi[0] = psi[0] / old_sum * remaining
                    psi[1] = psi[1] / old_sum * remaining
                else:
                    psi[0] = remaining / 2
                    psi[1] = remaining / 2
                grid_ss[r, c] = psi
            
            # 阴阳失衡修复 (即使和合态OK，阴阳严重失衡也要调)
            elif abs(psi[0] - psi[1]) > 0.3:
                w[2] += 0.1  # 补土
                w /= w.sum()
                grid_wx[r, c] = w
                
                avg = (psi[0] + psi[1]) / 2
                psi[0] = 0.8 * psi[0] + 0.2 * avg
                psi[1] = 0.8 * psi[1] + 0.2 * avg
                psi /= psi.sum()
                grid_ss[r, c] = psi
    
    return grid_ss, grid_wx


def inject_fault(grid_ss):
    """故障注入"""
    for r, c in FAULT_CELLS:
        grid_ss[r, c] = np.array([0.95, 0.03, 0.02])
    return grid_ss


# ============================================================
# 5. 初始化
# ============================================================
def initialize_grid():
    grid_ss = np.zeros((N, N, 3))
    grid_wx = np.zeros((N, N, 5))
    for r in range(N):
        for c in range(N):
            grid_ss[r, c] = np.random.dirichlet([2.5, 2.5, 3.5])
            grid_wx[r, c] = np.random.dirichlet([3, 1.5, 2, 1, 2.5])
    return grid_ss, grid_wx


# ============================================================
# 6. 验证指标
# ============================================================
def check_simplex(grid_ss, grid_wx):
    return np.max(np.abs(grid_ss.sum(2) - 1)), np.max(np.abs(grid_wx.sum(2) - 1))

def balance_metric(grid_ss):
    return np.mean(np.abs(grid_ss[:, :, 0] - grid_ss[:, :, 1]))

def harmony_metric(grid_ss):
    return np.mean(grid_ss[:, :, 2])

def wuxing_uniformity(grid_wx):
    avg = grid_wx.reshape(-1, 5).mean(0)
    return -np.sum(avg * np.log(avg + 1e-15)) / np.log(5)

def task_complexity(grid_ss):
    imb = balance_metric(grid_ss)
    return imb / (imb + 0.5)


# ============================================================
# 7. 主仿真
# ============================================================
def run():
    print("=" * 60)
    print("三生叠加态原生AGI架构 —— 最小可运行仿真 v1.2")
    print("=" * 60)
    print(f"网格: {N}×{N} 环面 | 步数: {STEPS}")
    print(f"ε={EPSILON} | λ_h={LAMBDA_H} | λ_bal={LAMBDA_BAL} | θ={THETA_GROW}")
    print()

    grid_ss, grid_wx = initialize_grid()
    
    hist = {k: [] for k in ['balance', 'harmony', 'wuxing_u', 'complexity', 'ss_err', 'wx_err']}
    snapshots = {}

    for t in range(STEPS):
        if t == FAULT_STEP:
            print(f"[Step {t}] ⚡ 故障注入: 中心4元胞极端失衡")
            grid_ss = inject_fault(grid_ss)

        grid_ss = update_sansheng(grid_ss)
        grid_wx = update_wuxing_local(grid_wx, grid_ss)
        grid_ss, grid_wx = organ_self_healing(grid_ss, grid_wx, fault_cells_active=(t >= FAULT_STEP))

        se, we = check_simplex(grid_ss, grid_wx)
        hist['ss_err'].append(se)
        hist['wx_err'].append(we)
        hist['balance'].append(balance_metric(grid_ss))
        hist['harmony'].append(harmony_metric(grid_ss))
        hist['wuxing_u'].append(wuxing_uniformity(grid_wx))
        hist['complexity'].append(task_complexity(grid_ss))

        if t in [0, 25, 50, 75, 100, 150, 199]:
            snapshots[t] = (grid_ss.copy(), grid_wx.copy())

        if t % 25 == 0:
            print(f"  Step {t:3d} | 平衡={hist['balance'][-1]:.4f} | "
                  f"和合={hist['harmony'][-1]:.4f} | "
                  f"五行={hist['wuxing_u'][-1]:.4f} | "
                  f"单纯形err={se:.2e}")

    print()
    print("=" * 60)
    print("✅ 验证结果:")
    print(f"  {'✅' if max(hist['ss_err']) < 1e-10 else '⚠️'} "
          f"三生单纯形守恒: max_err = {max(hist['ss_err']):.2e}")
    print(f"  {'✅' if max(hist['wx_err']) < 1e-10 else '⚠️'} "
          f"五行单纯形守恒: max_err = {max(hist['wx_err']):.2e}")

    b0, bf = hist['balance'][0], hist['balance'][-1]
    print(f"  {'✅' if bf <= b0 else '⚠️'} "
          f"平衡度收敛: {b0:.4f} → {bf:.4f} "
          f"({'↓ 收敛' if bf <= b0 else '↑ 发散'})")

    w0, wf = hist['wuxing_u'][0], hist['wuxing_u'][-1]
    print(f"  {'✅' if wf > 0.95 else '⚠️'} "
          f"五行均匀度: {w0:.4f} → {wf:.4f}")

    hp = hist['harmony'][FAULT_STEP - 1]
    hm = min(hist['harmony'][FAULT_STEP:FAULT_STEP + 20])
    hr = hist['harmony'][-1]
    print(f"  {'✅' if hr > hm + 0.02 else '⚠️'} "
          f"自愈恢复: 故障前={hp:.4f}, 最低={hm:.4f}, 恢复={hr:.4f}")
    print("=" * 60)

    return hist, snapshots


# ============================================================
# 8. 可视化
# ============================================================
def visualize(hist, snapshots):
    fig = plt.figure(figsize=(20, 28))
    gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3)

    steps = range(STEPS)

    # --- Row 0: 全局指标 ---
    ax = fig.add_subplot(gs[0, :])
    ax.plot(steps, hist['balance'], 'b-', lw=2, label='阴阳失衡度 |α-β|', alpha=0.85)
    ax.plot(steps, hist['harmony'], 'r-', lw=2, label='和合度 γ', alpha=0.85)
    ax.plot(steps, hist['wuxing_u'], 'g-', lw=2, label='五行均匀度', alpha=0.85)
    ax.plot(steps, hist['complexity'], 'purple', lw=1.5, label='任务复杂度 C̃', alpha=0.6, ls='--')
    ax.axvline(x=FAULT_STEP, color='orange', ls='--', lw=2.5, label=f'⚡ 故障注入 (step {FAULT_STEP})')
    ax.axvspan(FAULT_STEP, STEPS - 1, alpha=0.04, color='green')
    ax.set_xlabel('时间步', fontsize=13)
    ax.set_ylabel('指标值', fontsize=13)
    ax.set_title('三生叠加态仿真 v1.2 —— 全局指标演化', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, STEPS - 1)

    # --- Row 1: 单纯形误差 + 自愈详情 ---
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.semilogy(steps, np.array(hist['ss_err']) + 1e-18, 'b-', lw=1, label='三生 Δ²')
    ax1.semilogy(steps, np.array(hist['wx_err']) + 1e-18, 'r-', lw=1, label='五行 Δ⁴')
    ax1.axvline(x=FAULT_STEP, color='orange', ls='--', lw=2)
    ax1.set_title('单纯形守恒误差', fontsize=12)
    ax1.set_xlabel('时间步')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1])
    window = slice(max(0, FAULT_STEP - 10), min(STEPS, FAULT_STEP + 50))
    t_r = list(range(*window.indices(STEPS)))
    ax2.plot(t_r, hist['harmony'][window], 'r-o', ms=3, lw=1.5, label='和合度 γ')
    ax2.plot(t_r, hist['balance'][window], 'b-s', ms=3, lw=1.5, label='失衡度 |α-β|')
    ax2.axvline(x=FAULT_STEP, color='orange', ls='--', lw=2, label='⚡ 故障')
    ax2.set_title('故障→自愈恢复曲线', fontsize=12)
    ax2.set_xlabel('时间步')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 2])
    # 各元胞的和合度分布随时间变化
    ax3.plot(steps, hist['harmony'], 'r-', lw=2)
    ax3.fill_between(steps, 0, hist['harmony'], alpha=0.15, color='red')
    ax3.axvline(x=FAULT_STEP, color='orange', ls='--', lw=2)
    ax3.axhline(y=THETA_GROW, color='gray', ls=':', lw=1, label=f'阈值 θ={THETA_GROW}')
    ax3.set_title('全局平均和合度', fontsize=12)
    ax3.set_xlabel('时间步')
    ax3.set_ylabel('γ 均值')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Row 2-3: 三生单纯形快照 (6个时刻) ---
    snap_times = [0, 25, 50, 75, 100, 199]
    for idx, t_key in enumerate(snap_times):
        available = sorted(snapshots.keys())
        closest = min(available, key=lambda x: abs(x - t_key))
        grid_ss, _ = snapshots[closest]

        row = 2 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        a = grid_ss[:, :, 0].flatten()
        b = grid_ss[:, :, 1].flatten()
        g = grid_ss[:, :, 2].flatten()

        x = b + g / 2
        y = g * np.sqrt(3) / 2

        sc = ax.scatter(x, y, c=g, cmap='RdYlBu_r', s=100,
                        edgecolors='black', lw=0.5, vmin=0, vmax=max(0.5, g.max()))

        tri_x = [0, 1, 0.5, 0]
        tri_y = [0, 0, np.sqrt(3) / 2, 0]
        ax.plot(tri_x, tri_y, 'k-', lw=2)
        ax.text(-0.05, -0.05, 'α(阴)', fontsize=9, ha='right')
        ax.text(1.05, -0.05, 'β(阳)', fontsize=9, ha='left')
        ax.text(0.5, np.sqrt(3) / 2 + 0.05, 'γ(和合)', fontsize=9, ha='center')
        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(-0.1, 1.05)
        ax.set_aspect('equal')
        ax.set_axis_off()

        label = f'Step {closest}'
        if closest == FAULT_STEP:
            label += ' (故障后)'
        elif closest == 0:
            label += ' (初始)'
        elif closest == 199:
            label += ' (最终)'
        ax.set_title(label, fontsize=12, fontweight='bold')

    # --- Row 4: 五行主导热力图 ---
    heat_times = [0, 50, 199]
    for idx, t_key in enumerate(heat_times):
        available = sorted(snapshots.keys())
        closest = min(available, key=lambda x: abs(x - t_key))
        _, grid_wx = snapshots[closest]

        ax = fig.add_subplot(gs[4, idx])
        dominant = np.argmax(grid_wx, axis=2)

        from matplotlib.colors import ListedColormap
        colors = ['#4CAF50', '#F44336', '#FF9800', '#9E9E9E', '#2196F3']
        cmap = ListedColormap(colors)
        im = ax.imshow(dominant, cmap=cmap, vmin=-0.5, vmax=4.5)
        ax.set_title(f'五行主导分布 (Step {closest})', fontsize=12)
        ax.set_xlabel('列')
        ax.set_ylabel('行')
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(WUXING_NAMES)

    fig.suptitle('三生叠加态原生AGI架构 — 最小可运行仿真 v1.2',
                 fontsize=17, fontweight='bold', y=0.99)

    out = '/app/data/所有对话/主对话/simulation_results_v1.2.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n📊 可视化: {out}")
    return out


# ============================================================
# 9. 运行
# ============================================================
if __name__ == '__main__':
    h, s = run()
    img = visualize(h, s)
    print("✅ 仿真 v1.2 完成!")
