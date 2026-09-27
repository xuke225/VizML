#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器学习算法可视化平台 - 智能启动脚本
提供环境管理、依赖检查和自动浏览器启动功能
"""

import sys
import os
import subprocess
import argparse
import webbrowser
import time
import platform
from pathlib import Path

# 解决Windows控制台编码问题
if platform.system() == "Windows":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class MLPlatformLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.venv_path = self.project_root / '.venv'
        self.requirements_file = self.project_root / 'requirements.txt'
        self.app_file = self.project_root / 'app.py'
        
    def check_python_version(self):
        """检查Python版本"""
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            print("[错误] 需要Python 3.8或更高版本")
            print(f"当前版本: {sys.version}")
            return False
        print(f"[正确] Python版本: {sys.version.split()[0]}")
        return True
    
    def check_virtual_environment(self):
        """检查并创建虚拟环境"""
        if not self.venv_path.exists():
            print("[配置] 创建虚拟环境...")
            try:
                subprocess.run([sys.executable, '-m', 'venv', str(self.venv_path)], 
                             check=True, capture_output=True)
                print("[正确] 虚拟环境创建成功")
                return True
            except subprocess.CalledProcessError as e:
                print(f"[错误] 虚拟环境创建失败: {e}")
                return False
        else:
            print("[正确] 虚拟环境已存在")
            return True
    
    def get_venv_python(self):
        """获取虚拟环境Python路径（兼容标准 venv 与 conda 环境结构）"""
        if platform.system() == "Windows":
            if (self.venv_path / 'Scripts' / 'python.exe').exists():
                return self.venv_path / 'Scripts' / 'python.exe'
            return self.venv_path / 'python.exe'  # conda 结构
        else:
            if (self.venv_path / 'bin' / 'python').exists():
                return self.venv_path / 'bin' / 'python'
            return self.venv_path / 'python'  # conda 结构
    
    def get_venv_pip(self):
        """获取虚拟环境pip路径"""
        if platform.system() == "Windows":
            return self.venv_path / 'Scripts' / 'pip.exe'
        else:
            return self.venv_path / 'bin' / 'pip'
    
    def install_dependencies(self):
        """安装依赖包"""
        if not self.requirements_file.exists():
            print("[错误] requirements.txt文件不存在")
            return False
        
        print("📦 安装依赖包...")
        pip_path = self.get_venv_pip()
        
        try:
            subprocess.run([str(pip_path), 'install', '-r', str(self.requirements_file)], 
                         check=True)
            print("[正确] 依赖包安装完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[错误] 依赖包安装失败: {e}")
            return False
    
    def setup_environment(self):
        """完整环境设置"""
        print("🚀 开始环境设置...")
        
        if not self.check_python_version():
            return False
        
        if not self.check_virtual_environment():
            return False
        
        if not self.install_dependencies():
            return False
        
        print("[正确] 环境设置完成!")
        return True
    
    def show_system_info(self):
        """显示系统信息"""
        print("📊 系统信息:")
        print(f"   操作系统: {platform.system()} {platform.release()}")
        print(f"   Python版本: {sys.version.split()[0]}")
        print(f"   项目路径: {self.project_root}")
        print(f"   虚拟环境: {'存在' if self.venv_path.exists() else '不存在'}")
        
        # 检查依赖包
        pip_path = self.get_venv_pip()
        if pip_path.exists():
            try:
                result = subprocess.run([str(pip_path), 'list'], 
                                      capture_output=True, text=True)
                packages = result.stdout.strip().split('\n')[2:]  # 跳过标题行
                print(f"   已安装包数量: {len(packages)}")
            except:
                print("   已安装包数量: 无法获取")
    
    def start_server(self, debug=False, port=5432):
        """启动Flask服务器"""
        python_path = self.get_venv_python()
        
        if not python_path.exists():
            print("[错误] 虚拟环境不存在，请先运行: python start.py --setup")
            return False
        
        print(f"🌐 启动服务器 (端口: {port}, 调试模式: {debug})...")
        
        # 设置环境变量
        env = os.environ.copy()
        env['FLASK_APP'] = str(self.app_file)
        env['FLASK_ENV'] = 'development' if debug else 'production'
        
        try:
            # 延迟打开浏览器
            if not debug:
                import threading
                def open_browser():
                    time.sleep(2)  # 等待服务器启动
                    webbrowser.open(f'http://localhost:{port}')
                
                threading.Thread(target=open_browser, daemon=True).start()
            
            # 构建启动命令
            cmd = [str(python_path), str(self.app_file)]
            if port != 5432:
                env['PORT'] = str(port)
            
            print(f"[正确] 服务器启动成功!")
            print(f"📱 访问地址: http://localhost:{port}")
            print("🛑 按 Ctrl+C 停止服务器")
            
            subprocess.run(cmd, env=env)
            
        except KeyboardInterrupt:
            print("\n🛑 服务器已停止")
        except Exception as e:
            print(f"[错误] 服务器启动失败: {e}")
            return False
        
        return True

def main():
    parser = argparse.ArgumentParser(description='机器学习算法可视化平台启动脚本')
    parser.add_argument('--setup', action='store_true', 
                       help='完整环境设置（创建虚拟环境并安装依赖）')
    parser.add_argument('--install', action='store_true', 
                       help='仅安装依赖包')
    parser.add_argument('--info', action='store_true', 
                       help='显示系统信息')
    parser.add_argument('--debug', action='store_true', 
                       help='以调试模式启动')
    parser.add_argument('--port', type=int, default=5432,
                       help='指定服务器端口 (默认: 5432)')
    
    args = parser.parse_args()
    launcher = MLPlatformLauncher()
    
    if args.setup:
        launcher.setup_environment()
    elif args.install:
        launcher.install_dependencies()
    elif args.info:
        launcher.show_system_info()
    else:
        # 默认启动服务器
        # 如果虚拟环境不存在，先设置环境
        if not launcher.venv_path.exists():
            print("🔧 首次运行，正在设置环境...")
            if not launcher.setup_environment():
                sys.exit(1)
        
        launcher.start_server(debug=args.debug, port=args.port)

if __name__ == '__main__':
    main()
