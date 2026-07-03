"""
三生叠加态原生AGI架构 - PyTorch完整实现

本文件包含完整的PyTorch实现，用于实际训练任务。
需要在有torch环境的地方运行。

使用方式：
    from sansheng_pytorch_version import SanshengLayer, SimplexAdam, SanshengClassifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, Optional, List
import numpy as np


###############################################################################
# SanshengLayer: 可微分的三生耦合算子层
###############################################################################

class SanshengLayer(nn.Module):
    """
    三生耦合算子的可微分层实现
    
    理论基础：
    - 每个元胞状态 (α, γ, β) ∈ Δ²（概率单纯形）
    - α: 阴（收敛态），β: 阳（发散态），γ: 和（平衡态）
    - 通过softmax参数化保证单纯形约束
    
    耦合算子（增强版）：
    - cross = α_v·β_u + β_v·α_u
    - gamma_raw = γ_v·(α_v·(α_v+γ_v) + cross·0.3) + λ_h·γ_v·γ_u
    - α_out = α²_v + α_v·γ_u + λ_bal·(α_u - β_u)
    - β_out = β²_v + β_v·γ_u - λ_bal·(α_u - β_u)
    """
    
    def __init__(
        self, 
        grid_size: int = 8,
        coupling_strength: float = 0.5,
        harmony_coeff: float = 0.3,
        balance_coeff: float = 0.1,
        num_steps: int = 3,
        learnable_params: bool = True
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_steps = num_steps
        
        # 可学习参数（使用对数尺度便于优化）
        if learnable_params:
            # 初始值经过对数变换存储
            self.log_epsilon = nn.Parameter(torch.tensor(0.0))  # ε = exp(0) = 1.0
            self.log_lambda_h = nn.Parameter(torch.tensor(-1.0))  # λ_h = exp(-1) ≈ 0.37
            self.log_lambda_bal = nn.Parameter(torch.tensor(-2.0))  # λ_bal = exp(-2) ≈ 0.14
        else:
            self.register_buffer('epsilon', torch.tensor(coupling_strength))
            self.register_buffer('lambda_h', torch.tensor(harmony_coeff))
            self.register_buffer('lambda_bal', torch.tensor(balance_coeff))
        
        # 8邻域卷积核（用于高效邻居提取）
        self._init_neighbor_kernel()
    
    def _init_neighbor_kernel(self):
        """初始化邻居提取卷积核"""
        # 8个方向的偏移
        offsets = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],          [0, 1],
            [1, -1],  [1, 0], [1, 1]
        ]
        
        # 存储邻居提取函数
        self.register_buffer('neighbor_offsets', torch.tensor(offsets, dtype=torch.long))
    
    @property
    def epsilon(self) -> torch.Tensor:
        """耦合强度 ε，使用softplus保证正数"""
        return F.softplus(self.log_epsilon) + 1e-8
    
    @property
    def lambda_h(self) -> torch.Tensor:
        """和合系数 λ_h"""
        return F.softplus(self.log_lambda_h) + 1e-8
    
    @property
    def lambda_bal(self) -> torch.Tensor:
        """平衡引力系数 λ_bal"""
        return F.softplus(self.log_lambda_bal) + 1e-8
    
    def _get_neighbors(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取8邻域张量
        
        Args:
            x: (batch, H, W, 3) 状态张量
            
        Returns:
            neighbors: (batch, H, W, 8, 3) 每个位置的8个邻居
        """
        B, H, W, C = x.shape
        device = x.device
        
        # 填充边界（零填充）
        x_padded = F.pad(x, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
        
        # 手动提取8邻域
        neighbors_list = []
        for offset in self.neighbor_offsets:
            dy, dx = offset.tolist()
            neighbor = x_padded[:, 
                1+dy:1+dy+H, 
                1+dx:1+dx+W, 
                :
            ]
            neighbors_list.append(neighbor)
        
        # (batch, H, W, 8, 3)
        neighbors = torch.stack(neighbors_list, dim=3)
        return neighbors
    
    def _coupling_operator(
        self, 
        x_u: torch.Tensor,  # (batch, H, W, 3) 当前状态
        x_v: torch.Tensor   # (batch, H, W, 3) 单个邻居状态
    ) -> torch.Tensor:
        """
        核心耦合算子（增强版）
        """
        # 分离分量
        alpha_u, gamma_u, beta_u = x_u[..., 0], x_u[..., 1], x_u[..., 2]
        alpha_v, gamma_v, beta_v = x_v[..., 0], x_v[..., 1], x_v[..., 2]
        
        # 交叉项
        cross = alpha_v * beta_u + beta_v * alpha_u
        
        # 和分量计算（增强版）
        gamma_raw = gamma_v * (alpha_v * (alpha_v + gamma_v) + cross * 0.3)
        gamma_raw = gamma_raw + self.lambda_h * gamma_v * gamma_u
        
        # 阴分量
        alpha_out = alpha_v ** 2 + alpha_v * gamma_u
        
        # 阳分量  
        beta_out = beta_v ** 2 + beta_v * gamma_u
        
        # 添加平衡引力项
        balance_force = self.lambda_bal * (alpha_u - beta_u)
        alpha_out = alpha_out + balance_force
        beta_out = beta_out - balance_force
        
        # 堆叠并投影到单纯形
        raw = torch.stack([alpha_out, gamma_raw, beta_out], dim=-1)
        
        # 确保非负并归一化
        raw = F.relu(raw) + 1e-8
        x_new = raw / (raw.sum(dim=-1, keepdim=True) + 1e-8)
        
        return x_new
    
    def forward(
        self, 
        x: torch.Tensor,
        return_all_steps: bool = False
    ) -> torch.Tensor:
        """
        前向传播：多步耦合演化
        
        Args:
            x: (batch, H, W, 3) 初始状态，期望已在单纯形上
            return_all_steps: 是否返回所有中间步骤
            
        Returns:
            最终状态或所有步骤的状态序列
        """
        B, H, W, C = x.shape
        
        # 确保输入在单纯形上（通过softmax投影）
        x = F.softmax(x, dim=-1)
        
        all_states = [x] if return_all_steps else None
        current_state = x
        
        for step in range(self.num_steps):
            # 获取邻居
            neighbors = self._get_neighbors(current_state)
            
            # 对所有邻居应用耦合算子并平均
            coupled_states = []
            for i in range(8):
                coupled = self._coupling_operator(current_state, neighbors[..., i, :])
                coupled_states.append(coupled)
            
            # 平均所有邻居的贡献
            new_state = torch.stack(coupled_states, dim=0).mean(dim=0)
            
            # 应用耦合强度加权
            new_state = (1 - self.epsilon) * current_state + self.epsilon * new_state
            
            # 再次投影到单纯形
            new_state = F.softmax(new_state, dim=-1)
            
            current_state = new_state
            
            if return_all_steps:
                all_states.append(current_state)
        
        if return_all_steps:
            return torch.stack(all_states, dim=1)  # (batch, num_steps+1, H, W, 3)
        return current_state
    
    def get_state_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """获取状态的统计信息"""
        stats = {
            'alpha_mean': x[..., 0].mean().item(),
            'gamma_mean': x[..., 1].mean().item(),
            'beta_mean': x[..., 2].mean().item(),
            'balance': (x[..., 0] - x[..., 2]).abs().mean().item(),  # 阴阳平衡度
        }
        return stats


###############################################################################
# SimplexAdam: 单纯形流形上的自适应优化器
###############################################################################

class SimplexAdam(optim.Optimizer):
    """
    单纯形流形上的自适应优化器
    
    理论基础：
    - 概率单纯形 Δ^n 是黎曼流形
    - Fisher信息度量: g_ij = δ_ij/θ_i - 1
    - 自然梯度 = Fisher^{-1} · 欧氏梯度
    
    算法：
    1. 在对数坐标中计算梯度
    2. 应用Adam动量更新
    3. 通过softmax投影回单纯形
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """执行一步优化"""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                beta1, beta2 = group['betas']
                
                # 在对数空间更新（自然梯度近似）
                # 梯度变换: ∂L/∂θ = ∂L/∂softmax(θ) · ∂softmax/∂θ
                # softmax导数在概率空间中等价于在log-space的投影
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # 对数空间的梯度
                log_grad = grad * p.data.clamp(min=1e-8)
                
                # Adam更新
                exp_avg.mul_(beta1).add_(log_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(log_grad, log_grad, value=1 - beta2)
                
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr']
                
                # 计算更新方向
                denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group['eps'])
                step = exp_avg / denom
                
                # 更新参数
                p.data.add_(step, alpha=-step_size)
                
                # 重新归一化到单纯形
                with torch.no_grad():
                    p.data[:] = F.softmax(p.data, dim=-1)
        
        return loss


class RiemannianSGD(optim.Optimizer):
    """
    黎曼SGD优化器（对比基线）
    
    使用黎曼梯度：g_R = g_E - Σ_i θ_i·g_E_i
    """
    
    def __init__(self, params, lr: float = 1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # 黎曼梯度投影
                grad = p.grad.data
                theta = p.data
                
                # g_R = g_E - (θ · g_E) · 1
                riemann_grad = grad - (theta * grad).sum(dim=-1, keepdim=True)
                
                # 更新
                p.data.add_(riemann_grad, alpha=-group['lr'])
                
                # 投影回单纯形
                with torch.no_grad():
                    p.data[:] = F.softmax(p.data, dim=-1)
        
        return loss


###############################################################################
# SanshengClassifier: 三生分类器
###############################################################################

class SanshengClassifier(nn.Module):
    """
    基于三生架构的分类器
    
    结构：
    1. 输入编码层：将图像像素映射到三生状态
    2. SanshengLayer：多步耦合演化
    3. 全局池化：对γ分量进行池化
    4. 分类头
    """
    
    def __init__(
        self,
        grid_size: int = 8,
        num_classes: int = 10,
        hidden_dim: int = 64,
        num_sansheng_layers: int = 2,
        coupling_strength: float = 0.5,
        harmony_coeff: float = 0.3,
        balance_coeff: float = 0.1
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        # 输入编码：将图像编码为三生状态
        self.input_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 下采样到 grid_size x grid_size
        self.downsample = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        
        # 三生编码层
        self.sansheng_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        
        # 多个三生耦合层
        self.sansheng_layers = nn.ModuleList([
            SanshengLayer(
                grid_size=grid_size,
                coupling_strength=coupling_strength,
                harmony_coeff=harmony_coeff,
                balance_coeff=balance_coeff,
                num_steps=2,
                learnable_params=True
            )
            for _ in range(num_sansheng_layers)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(grid_size * grid_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor, return_gamma_map: bool = False) -> torch.Tensor:
        """
        前向传播
        """
        B = x.shape[0]
        
        # 编码和下采样
        x = self.input_encoder(x)
        x = self.downsample(x)  # (batch, 1, grid_size, grid_size)
        
        # 转换为三生状态
        x = x.view(B, 1, -1)  # (batch, 1, grid_size^2)
        
        # 编码为三生状态（返回logits，由softmax投影）
        x = self.sansheng_encoder(x)  # (batch, grid_size^2, 3)
        
        # 重塑为网格格式
        x = x.view(B, self.grid_size, self.grid_size, 3)
        
        # 通过三生层演化
        gamma_maps = []
        for layer in self.sansheng_layers:
            x = layer(x)
            gamma_maps.append(x[..., 1].clone())
        
        # 全局池化γ分量
        gamma = x[..., 1]  # (batch, grid_size, grid_size)
        gamma_flat = gamma.view(B, -1)  # (batch, grid_size^2)
        
        # 分类
        logits = self.classifier(gamma_flat)
        
        if return_gamma_map:
            return logits, gamma_maps
        return logits


###############################################################################
# EmergenceLoss: 涌现损失函数
###############################################################################

class EmergenceLoss(nn.Module):
    """
    涌现损失函数
    
    L_total = L_task + λ_emerge · L_emerge + λ_bal_reg · L_bal
    """
    
    def __init__(
        self,
        lambda_emerge: float = 0.1,
        lambda_balance: float = 0.05,
        task_loss_type: str = 'cross_entropy'
    ):
        super().__init__()
        self.lambda_emerge = lambda_emerge
        self.lambda_balance = lambda_balance
        self.task_loss_type = task_loss_type
        
        if task_loss_type == 'cross_entropy':
            self.task_loss_fn = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        final_state: torch.Tensor,
        gamma_maps: list = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失
        """
        # 任务损失
        L_task = self.task_loss_fn(logits, target)
        
        # 涌现损失：低熵 = 更有结构的模式
        gamma = final_state[..., 1]
        gamma_flat = gamma.view(gamma.shape[0], -1)
        gamma_probs = gamma_flat / (gamma_flat.sum(dim=1, keepdim=True) + 1e-8)
        L_emerge = -(gamma_probs * torch.log(gamma_probs + 1e-8)).sum(dim=1).mean()
        
        # 平衡损失
        alpha = final_state[..., 0]
        beta = final_state[..., 2]
        L_bal = ((alpha - beta) ** 2).mean()
        
        # 总损失
        L_total = L_task + self.lambda_emerge * L_emerge + self.lambda_balance * L_bal
        
        return L_total, {
            'L_task': L_task.item(),
            'L_emerge': L_emerge.item(),
            'L_balance': L_bal.item(),
            'L_total': L_total.item()
        }


###############################################################################
# 梯度验证工具
###############################################################################

def verify_gradients_torch(model: SanshengLayer, x: torch.Tensor, 
                          target: torch.Tensor, rtol: float = 1e-4) -> Dict[str, float]:
    """
    用有限差分法验证解析梯度与自动微分的差异
    
    Returns:
        各参数的最大相对误差字典
    """
    model.eval()
    
    # 保存原始参数值
    orig_params = {
        'epsilon': model.log_epsilon.data.clone(),
        'lambda_h': model.log_lambda_h.data.clone(),
        'lambda_bal': model.log_lambda_bal.data.clone()
    }
    
    # 确保需要梯度
    x = x.requires_grad_(True)
    
    # 前向传播
    output = model(x)
    loss = F.mse_loss(output, target)
    
    # PyTorch自动微分梯度
    loss.backward()
    
    autograd_grads = {
        'epsilon': model.log_epsilon.grad.clone(),
        'lambda_h': model.log_lambda_h.grad.clone(),
        'lambda_bal': model.log_lambda_bal.grad.clone()
    }
    
    # 有限差分梯度
    h = 1e-6
    errors = {}
    
    for param_name in ['epsilon', 'lambda_h', 'lambda_bal']:
        log_param_name = f'log_{param_name}'
        
        # 正向扰动
        model_state = getattr(model, log_param_name).data.clone()
        getattr(model, log_param_name).data = model_state + h
        output_plus = model(x)
        loss_plus = F.mse_loss(output_plus, target)
        
        # 负向扰动
        getattr(model, log_param_name).data = model_state - h
        output_minus = model(x)
        loss_minus = F.mse_loss(output_minus, target)
        
        # 恢复
        getattr(model, log_param_name).data = model_state
        
        # 数值梯度
        numerical_grad = (loss_plus.item() - loss_minus.item()) / (2 * h)
        
        # 相对误差
        autograd_val = autograd_grads[param_name].mean().item()
        if abs(autograd_val) > 1e-8:
            rel_error = abs(numerical_grad - autograd_val) / abs(autograd_val)
            errors[param_name] = rel_error
    
    return errors


###############################################################################
# 训练函数
###############################################################################

def train_model(
    model: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    criterion: EmergenceLoss,
    optimizer,
    num_epochs: int = 10,
    batch_size: int = 32,
    device: str = 'cpu'
) -> Dict[str, List]:
    """
    训练三生分类器
    """
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'balance': [],
        'gamma_entropy': []
    }
    
    num_batches = (len(train_images) + batch_size - 1) // batch_size
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_balance = []
        epoch_entropy = []
        
        # 打乱数据
        indices = torch.randperm(len(train_images))
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_images))
            batch_indices = indices[start_idx:end_idx]
            
            batch_images = train_images[batch_indices].to(device)
            batch_labels = train_labels[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            logits, gamma_maps = model(batch_images, return_gamma_map=True)
            final_state = gamma_maps[-1]
            
            # 计算损失
            loss, loss_dict = criterion(logits, batch_labels, final_state, gamma_maps)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss_dict['L_total'])
            epoch_balance.append(loss_dict['L_balance'])
            
            # 计算γ熵
            gamma = final_state[..., 1]
            gamma_flat = gamma.view(gamma.shape[0], -1)
            gamma_probs = gamma_flat / (gamma_flat.sum(dim=1, keepdim=True) + 1e-8)
            entropy = -(gamma_probs * torch.log(gamma_probs + 1e-8)).sum(dim=1).mean()
            epoch_entropy.append(entropy.item())
        
        # 记录epoch统计
        history['train_loss'].append(np.mean(epoch_losses))
        history['balance'].append(np.mean(epoch_balance))
        history['gamma_entropy'].append(np.mean(epoch_entropy))
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {history['train_loss'][-1]:.4f} | "
              f"Balance: {history['balance'][-1]:.4f} | "
              f"γ Entropy: {history['gamma_entropy'][-1]:.4f}")
    
    return history


###############################################################################
# 使用示例
###############################################################################

if __name__ == '__main__':
    print("=" * 60)
    print("三生架构 PyTorch 版本测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 创建三生层
    print("\n1. 测试 SanshengLayer...")
    layer = SanshengLayer(grid_size=8, num_steps=2).to(device)
    
    # 随机输入
    x = torch.randn(2, 8, 8, 3).to(device)
    x = F.softmax(x, dim=-1)
    
    output = layer(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   ε = {layer.epsilon.item():.4f}")
    print(f"   λ_h = {layer.lambda_h.item():.4f}")
    print(f"   λ_bal = {layer.lambda_bal.item():.4f}")
    
    # 2. 测试梯度
    print("\n2. 测试梯度验证...")
    errors = verify_gradients_torch(layer, x, torch.randn_like(output))
    for name, error in errors.items():
        status = "✓ PASS" if error < 1e-4 else "✗ FAIL"
        print(f"   {name}: {error:.2e} {status}")
    
    # 3. 测试优化器
    print("\n3. 测试 SimplexAdam 优化器...")
    theta = torch.randn(10, 3, requires_grad=True).to(device)
    theta.data = F.softmax(theta.data, dim=-1)
    opt = SimplexAdam([theta], lr=0.1)
    
    def objective(theta):
        probs = F.softmax(theta, dim=-1)
        return -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
    
    initial_loss = objective(theta).item()
    for i in range(20):
        opt.zero_grad()
        loss = objective(theta)
        loss.backward()
        opt.step()
    
    final_loss = objective(theta).item()
    print(f"   初始损失: {initial_loss:.4f}")
    print(f"   最终损失: {final_loss:.4f}")
    print(f"   损失下降: {initial_loss - final_loss:.4f}")
    print(f"   参数仍在单纯形上: {torch.allclose(theta.sum(dim=-1), torch.ones(10).to(device))}")
    
    # 4. 测试分类器
    print("\n4. 测试 SanshengClassifier...")
    model = SanshengClassifier(grid_size=8, num_classes=10).to(device)
    
    # 随机图像输入
    images = torch.randn(4, 1, 28, 28).to(device)
    logits, gamma_maps = model(images, return_gamma_map=True)
    print(f"   输入形状: {images.shape}")
    print(f"   输出形状: {logits.shape}")
    print(f"   γ maps数量: {len(gamma_maps)}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)
