@echo off
chcp 65001 >nul
echo ========================================
echo   跌倒检测模型训练 - RTX 3060 GPU
echo ========================================

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 未安装！请安装 Python 3.10+
    pause
    exit /b 1
)

REM 安装依赖（首次运行会慢一些）
echo [1/3] 检查并安装依赖...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install scikit-learn scipy tqdm pyyaml python-dotenv timm einops -q

REM 开始训练
echo.
echo [2/3] 开始训练...
echo [3/3] 模型: GaitLSTM ^| 数据: NTU ^| 设备: CUDA ^| Epochs: 100
echo.
python scripts/train.py --model gait_lstm --dataset ntu --device cuda:0 --epochs 100 --batch-size 32 --lr 0.001

echo.
echo 训练完成！模型保存在 checkpoints\ 目录
pause
