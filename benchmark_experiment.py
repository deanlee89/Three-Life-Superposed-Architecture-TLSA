#!/usr/bin/env python3
"""
三生架构4层 - MNIST/CIFAR-10 基准测试与对比实验

包含：
1. 4层架构的 PyTorch GPU 实现
2. 数据加载与预处理
3. 对比模型（MLP、CNN、标准张量网络）
4. 完整的训练和评估流程
5. 性能对比与结果分析
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Dict, List
import json
from datetime import datetime

# 设置GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


# ==================== 数据加载 ====================

class DataManager:
    """数据管理器"""
    
    def __init__(self, dataset_name: str = 'mnist', batch_size: int = 128, use_cuda: bool = False):
        """
        参数：
            dataset_name: 'mnist' 或 'cifar10'
            batch_size: 批大小
            use_cuda: 如果True，则启用pin_memory加速GPU数据加载
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.num_classes = 10
        self.input_dim = 784 if dataset_name == 'mnist' else 3072
        
        # 数据转换
        if dataset_name == 'mnist':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:  # cifar10
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None
    
    def load_data(self):
        """加载数据集"""
        if self.dataset_name == 'mnist':
            self._load_mnist()
        elif self.dataset_name == 'cifar10':
            self._load_cifar10()
    
    def _load_mnist(self):
        """加载MNIST"""
        print("加载MNIST数据集...")
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, transform=self.transform, download=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, transform=self.transform, download=True
        )
        
        # 分割训练集为训练和验证
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=self.use_cuda
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.use_cuda
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.use_cuda
        )
        
        print(f"✅ MNIST加载完成")
        print(f"   训练集: {len(self.train_loader)*self.batch_size}")
        print(f"   验证集: {len(self.val_loader)*self.batch_size}")
        print(f"   测试集: {len(self.test_loader)*self.batch_size}")
    
    def _load_cifar10(self):
        """加载CIFAR-10"""
        print("加载CIFAR-10数据集...")
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, transform=self.transform, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, transform=self.transform, download=True
        )
        
        # 分割训练集
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=self.use_cuda
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.use_cuda
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=self.use_cuda
        )
        
        print(f"✅ CIFAR-10加载完成")
        print(f"   训练集: {len(self.train_loader)*self.batch_size}")
        print(f"   验证集: {len(self.val_loader)*self.batch_size}")
        print(f"   测试集: {len(self.test_loader)*self.batch_size}")


# ==================== 模型定义 ====================

class Sansheng4LayerPyTorch(nn.Module):
    """三生架构4层 - PyTorch GPU实现"""
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 第1层：编码-路由
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Softmax(dim=1)
        )
        
        # 第2层：三生耦合-法则
        self.coupling = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 第3层：拓扑多尺度
        self.topology = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 第4层：自组织-收敛
        self.selforg = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )
        
        # 分类头
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x):
        """前向传播"""
        # 展平输入
        x = x.view(x.size(0), -1)
        
        # 第1层：编码-路由
        e = self.encoder(x)
        route_prob = self.router(e)
        
        # 第2层：三生耦合-法则
        coupled = torch.cat([e, route_prob], dim=1)
        psi = self.coupling(coupled)
        
        # 第3层：拓扑多尺度
        Y = self.topology(psi)
        
        # 第4层：自组织-收敛
        Y_org = self.selforg(Y)
        
        # 分类
        logits = self.classifier(Y_org)
        
        return logits


class MLPBaseline(nn.Module):
    """MLP基准模型"""
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


class CNNBaseline(nn.Module):
    """CNN基准模型"""
    
    def __init__(self, num_classes: int = 10, is_cifar: bool = False):
        super().__init__()
        in_channels = 3 if is_cifar else 1
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算扁平化后的维度
        if is_cifar:
            self.flat_dim = 128 * 4 * 4  # 32×32 → 4×4
        else:
            self.flat_dim = 128 * 3 * 3  # 28×28 → 3×3
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TensorNetworkClassifier(nn.Module):
    """标准张量网络分类器"""
    
    def __init__(self, input_dim: int = 784, bond_dim: int = 32, num_classes: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.bond_dim = bond_dim
        self.num_classes = num_classes
        
        # 简化的张量网络分类器
        self.left_tensor = nn.Parameter(
            torch.randn(input_dim, bond_dim) / np.sqrt(input_dim)
        )
        self.middle_tensor = nn.Parameter(
            torch.randn(bond_dim, bond_dim) / np.sqrt(bond_dim)
        )
        self.classifier = nn.Linear(bond_dim, num_classes)
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # 投影到张量网络的bond空间
        hidden = x @ self.left_tensor
        hidden = torch.tanh(hidden)
        hidden = hidden @ self.middle_tensor
        logits = self.classifier(hidden)
        return logits


# ==================== 训练与评估 ====================

class Trainer:
    """训练器"""
    
    def __init__(self, model, device, dataset_name='mnist'):
        self.model = model.to(device)
        self.device = device
        self.dataset_name = dataset_name
        
        # 优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # 记录
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': None,
            'test_acc': None,
            'train_time': 0,
        }
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self, test_loader):
        """测试"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=20):
        """完整训练流程"""
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 调整学习率
            self.scheduler.step()
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"早停（Epoch {epoch+1}）")
                self.model.load_state_dict(best_model_state)
                break
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        self.history['train_time'] = time.time() - start_time


# ==================== 主程序 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description='三生架构4层 - MNIST/CIFAR-10 基准测试'
    )
    parser.add_argument('--dataset', choices=['mnist', 'cifar10', 'all'], default='all',
                        help='选择数据集: mnist, cifar10, 或 all')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='训练和测试批大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练总epoch数')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式: 只运行MNIST，最多5个epoch')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                        help='保存结果的JSON文件名')
    return parser.parse_args()


def main():
    args = parse_args()
    print("\n" + "="*70)
    print("三生架构4层 - MNIST/CIFAR-10 基准测试")
    print(f"使用设备: {DEVICE}")
    print("="*70 + "\n")
    
    if args.quick:
        datasets = ['mnist']
        epochs = min(args.epochs, 5)
        print("⚡ 快速模式: MNIST 数据集，最多 5 个 epoch")
    else:
        epochs = args.epochs
        if args.dataset == 'all':
            datasets = ['mnist', 'cifar10']
        else:
            datasets = [args.dataset]
    
    batch_size = args.batch_size
    results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"数据集: {dataset_name.upper()}")
        print(f"{'='*70}\n")
        
        # 加载数据
        data_manager = DataManager(dataset_name, batch_size, use_cuda=(DEVICE.type == 'cuda'))
        data_manager.load_data()
        
        results[dataset_name] = {}
        
        # 定义模型
        models_config = {
            'Sansheng-4Layer': {
                'model': Sansheng4LayerPyTorch(
                    input_dim=data_manager.input_dim,
                    hidden_dim=128,
                    num_classes=10
                ),
                'name': '三生架构4层'
            },
            'MLP': {
                'model': MLPBaseline(
                    input_dim=data_manager.input_dim,
                    hidden_dim=256,
                    num_classes=10
                ),
                'name': 'MLP基准'
            },
            'CNN': {
                'model': CNNBaseline(
                    num_classes=10,
                    is_cifar=(dataset_name == 'cifar10')
                ),
                'name': 'CNN基准'
            },
            'TensorNetwork': {
                'model': TensorNetworkClassifier(
                    input_dim=data_manager.input_dim,
                    bond_dim=32,
                    num_classes=10
                ),
                'name': '标准张量网络'
            }
        }
        
        # 训练所有模型
        for model_name, config in models_config.items():
            print(f"\n训练: {config['name']}")
            print("-" * 70)
            
            trainer = Trainer(config['model'], DEVICE, dataset_name)
            
            # 统计参数
            param_count = sum(p.numel() for p in config['model'].parameters())
            print(f"参数数量: {param_count:,}")
            
            # 训练
            trainer.train(
                data_manager.train_loader,
                data_manager.val_loader,
                epochs=epochs
            )
            
            # 测试
            test_loss, test_acc = trainer.test(data_manager.test_loader)
            
            # 记录结果
            results[dataset_name][model_name] = {
                'name': config['name'],
                'param_count': param_count,
                'train_loss': trainer.history['train_loss'][-1],
                'train_acc': trainer.history['train_acc'][-1],
                'val_loss': trainer.history['val_loss'][-1],
                'val_acc': trainer.history['val_acc'][-1],
                'test_loss': test_loss,
                'test_acc': test_acc,
                'train_time': trainer.history['train_time'],
            }
            
            print(f"\n✅ 完成")
            print(f"测试准确率: {test_acc:.2f}%")
            print(f"训练时间: {trainer.history['train_time']:.2f}s")
    
    # 生成报告
    print("\n" + "="*70)
    print("结果总结")
    print("="*70 + "\n")
    
    for dataset_name, models in results.items():
        print(f"\n{dataset_name.upper()} 数据集结果:\n")
        print(f"{'模型':<20} {'测试准确率':<12} {'参数数量':<12} {'训练时间':<12}")
        print("-" * 56)
        
        for model_name, metrics in models.items():
            print(f"{metrics['name']:<20} {metrics['test_acc']:<11.2f}% "
                  f"{metrics['param_count']:<11,} {metrics['train_time']:<11.2f}s")
    
    # 保存结果
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存到 {args.output}")


if __name__ == '__main__':
    main()

