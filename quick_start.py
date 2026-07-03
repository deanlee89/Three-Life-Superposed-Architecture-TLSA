#!/usr/bin/env python3
"""
三生架构4层 - 基准测试快速启动脚本

功能：
1. 环境检查
2. 依赖验证
3. 快速运行实验
4. 生成报告
"""

import sys
import subprocess
from pathlib import Path


def check_environment():
    """检查环境"""
    print("="*70)
    print("三生架构4层 - 基准测试快速启动")
    print("="*70)
    print("\n【第1步】环境检查\n")
    
    # 检查Python版本
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"✅ Python版本: {py_version}")
    
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8+")
        sys.exit(1)
    
    # 检查关键依赖
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} 已安装")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing.append(module)
    
    if missing:
        print(f"\n❌ 缺少依赖: {', '.join(missing)}")
        print("\n【解决方案】")
        print("运行以下命令安装依赖：\n")
        print("pip install torch torchvision torchaudio matplotlib seaborn")
        print("\n（如果需要GPU支持，请访问 https://pytorch.org/get-started ）")
        response = input("\n继续吗？(y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # 检查GPU
    print("\n【GPU支持】")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
        else:
            print("⚠️  GPU不可用，将使用CPU（速度可能较慢）")
    except:
        print("⚠️  无法检查GPU支持")
    
    print("\n✅ 环境检查完成\n")


def run_experiment(quick: bool = False):
    """运行实验"""
    print("="*70)
    print("【第2步】运行基准测试")
    print("="*70 + "\n")
    
    if quick:
        print("⚡ 快速模式: MNIST数据集, 5个epoch\n")
        cmd = [
            'python', 'benchmark_experiment.py',
            '--dataset', 'mnist',
            '--epochs', '5',
            '--quick'
        ]
    else:
        print("📊 完整模式: MNIST和CIFAR-10, 20个epoch\n")
        print("预期时间: 10-30分钟（取决于硬件）\n")
        cmd = ['python', 'benchmark_experiment.py']
    
    try:
        subprocess.run(cmd, check=True)
        print("\n✅ 基准测试完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 运行出错: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")
        return False


def generate_report():
    """生成报告"""
    print("="*70)
    print("【第3步】生成分析报告")
    print("="*70 + "\n")
    
    if not Path('benchmark_results.json').exists():
        print("❌ 找不到结果文件 benchmark_results.json")
        print("请先运行基准测试\n")
        return False
    
    try:
        subprocess.run(['python', 'analyze_benchmark.py'], check=True)
        print("\n✅ 报告生成完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 生成报告出错: {e}\n")
        return False


def show_results():
    """显示结果"""
    print("="*70)
    print("【第4步】查看结果")
    print("="*70 + "\n")
    
    results_file = Path('benchmark_results.json')
    if results_file.exists():
        print("📊 生成的文件：\n")
        
        files = [
            ('benchmark_results.json', '原始数据'),
            ('benchmark_text_report.txt', '文本报告'),
            ('benchmark_report.md', 'Markdown报告'),
            ('accuracy_comparison.png', '准确率对比'),
            ('param_efficiency.png', '参数效率'),
            ('speed_comparison.png', '训练速度'),
        ]
        
        for filename, description in files:
            path = Path(filename)
            if path.exists():
                size = path.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f}MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"✅ {filename:<30} ({size_str}) - {description}")
            else:
                print(f"❌ {filename:<30} - {description}")
        
        print("\n【查看报告】")
        print("- 文本版本: cat benchmark_text_report.txt")
        print("- Markdown: cat benchmark_report.md")
        print("- 图表: 使用图片查看器打开 *.png")
        
        return True
    else:
        print("❌ 尚未运行实验\n")
        return False


def show_menu():
    """显示菜单"""
    print("\n" + "="*70)
    print("选择操作")
    print("="*70)
    print("""
1. 【快速运行】MNIST数据集, 5个epoch (5分钟)
2. 【完整运行】所有数据集, 20个epoch (15-30分钟)
3. 【生成报告】从现有结果生成分析报告
4. 【查看结果】显示已生成的文件
5. 【退出】

    """)
    
    choice = input("请选择 (1-5): ").strip()
    return choice


def main():
    """主函数"""
    # 检查环境
    check_environment()
    
    # 菜单循环
    while True:
        choice = show_menu()
        
        if choice == '1':
            if run_experiment(quick=True):
                if generate_report():
                    show_results()
        
        elif choice == '2':
            if run_experiment(quick=False):
                if generate_report():
                    show_results()
        
        elif choice == '3':
            generate_report()
        
        elif choice == '4':
            show_results()
        
        elif choice == '5':
            print("\n👋 再见！")
            break
        
        else:
            print("❌ 无效选择，请重试\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断\n")
    except Exception as e:
        print(f"\n❌ 错误: {e}\n")

