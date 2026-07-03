"""
Three-Life (Sansheng) Architecture - PyTorch Implementation
Requires: torch, numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Tuple, Dict, Optional, List
import numpy as np


class SanshengLayer(nn.Module):
    """
    Differentiable Three-Life coupling operator layer.
    State: (alpha, gamma, beta) in Delta^2 (probability simplex)
    alpha: Yin (convergent), beta: Yang (divergent), gamma: Harmony (balance)
    """
    
    def __init__(self, grid_size=8, num_steps=3, learnable_params=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_steps = num_steps
        
        if learnable_params:
            self.log_epsilon = nn.Parameter(torch.tensor(0.0))
            self.log_lambda_h = nn.Parameter(torch.tensor(-1.0))
            self.log_lambda_bal = nn.Parameter(torch.tensor(-2.0))
        
        self.register_buffer('neighbor_offsets', torch.tensor([
            [-1,-1],[-1,0],[-1,1],
            [0,-1],[0,1],
            [1,-1],[1,0],[1,1]
        ], dtype=torch.long))
    
    @property
    def epsilon(self):
        return F.softplus(self.log_epsilon) + 1e-8
    
    @property
    def lambda_h(self):
        return F.softplus(self.log_lambda_h) + 1e-8
    
    @property
    def lambda_bal(self):
        return F.softplus(self.log_lambda_bal) + 1e-8
    
    def _get_neighbors(self, x):
        B, H, W, C = x.shape
        x_padded = F.pad(x, (0,0,1,1,1,1), mode='constant', value=0)
        nbrs = []
        for off in self.neighbor_offsets:
            dy, dx = off.tolist()
            nbrs.append(x_padded[:, 1+dy:1+dy+H, 1+dx:1+dx+W, :])
        return torch.stack(nbrs, dim=3)
    
    def _coupling(self, x_u, x_v):
        a_u, g_u, b_u = x_u[...,0], x_u[...,1], x_u[...,2]
        a_v, g_v, b_v = x_v[...,0], x_v[...,1], x_v[...,2]
        
        cross = a_v*b_u + b_v*a_u
        g_raw = g_v*(a_v*(a_v+g_v) + cross*0.3) + self.lambda_h*g_v*g_u
        a_out = a_v**2 + a_v*g_u + self.lambda_bal*(a_u-b_u)
        b_out = b_v**2 + b_v*g_u - self.lambda_bal*(a_u-b_u)
        
        raw = F.relu(torch.stack([a_out, g_raw, b_out], dim=-1)) + 1e-8
        return raw / (raw.sum(dim=-1, keepdim=True) + 1e-8)
    
    def forward(self, x, return_all_steps=False):
        x = F.softmax(x, dim=-1)
        states = [x] if return_all_steps else None
        cur = x
        
        for _ in range(self.num_steps):
            nbrs = self._get_neighbors(cur)
            coupled = [self._coupling(cur, nbrs[...,i,:]) for i in range(8)]
            new = torch.stack(coupled, dim=0).mean(dim=0)
            new = (1-self.epsilon)*cur + self.epsilon*new
            cur = F.softmax(new, dim=-1)
            if return_all_steps:
                states.append(cur)
        
        return torch.stack(states, dim=1) if return_all_steps else cur


class SimplexAdam(optim.Optimizer):
    """Adam optimizer on the probability simplex manifold."""
    
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                b1, b2 = group['betas']
                log_grad = p.grad.data * p.data.clamp(min=1e-8)
                
                state['exp_avg'].mul_(b1).add_(log_grad, alpha=1-b1)
                state['exp_avg_sq'].mul_(b2).addcmul_(log_grad, log_grad, value=1-b2)
                
                bc1 = 1 - b1**state['step']
                bc2 = 1 - b2**state['step']
                denom = (state['exp_avg_sq'].sqrt()/bc2**0.5).add_(group['eps'])
                p.data.add_(state['exp_avg']/denom, alpha=-group['lr'])
                
                with torch.no_grad():
                    p.data[:] = F.softmax(p.data, dim=-1)
        return loss


class SanshengClassifier(nn.Module):
    """Three-Life classifier for image classification."""
    
    def __init__(self, grid_size=8, num_classes=10, hidden_dim=64, num_layers=2):
        super().__init__()
        self.grid_size = grid_size
        
        self.input_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.ReLU()
        )
        self.downsample = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.ss_encoder = nn.Sequential(nn.Linear(1,16), nn.ReLU(), nn.Linear(16,3))
        
        self.ss_layers = nn.ModuleList([
            SanshengLayer(grid_size=grid_size, num_steps=2)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(grid_size**2, hidden_dim), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, return_gamma_map=False):
        B = x.shape[0]
        x = self.input_encoder(x)
        x = self.downsample(x).view(B, 1, -1)
        x = self.ss_encoder(x).view(B, self.grid_size, self.grid_size, 3)
        
        gamma_maps = []
        for layer in self.ss_layers:
            x = layer(x)
            gamma_maps.append(x[...,1].clone())
        
        logits = self.classifier(x[...,1].view(B, -1))
        return (logits, gamma_maps) if return_gamma_map else logits


class EmergenceLoss(nn.Module):
    """L_total = L_task + lambda_e * L_emerge + lambda_b * L_balance"""
    
    def __init__(self, lambda_emerge=0.1, lambda_balance=0.05):
        super().__init__()
        self.lambda_emerge = lambda_emerge
        self.lambda_balance = lambda_balance
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits, target, final_state):
        L_task = self.ce(logits, target)
        g = final_state[...,1].reshape(final_state.shape[0], -1)
        gp = g / (g.sum(dim=1, keepdim=True) + 1e-8)
        L_emerge = -(gp * torch.log(gp + 1e-8)).sum(dim=1).mean()
        L_bal = ((final_state[...,0] - final_state[...,2])**2).mean()
        L_total = L_task + self.lambda_emerge*L_emerge + self.lambda_balance*L_bal
        return L_total, {'task': L_task.item(), 'emerge': L_emerge.item(),
                        'balance': L_bal.item(), 'total': L_total.item()}
