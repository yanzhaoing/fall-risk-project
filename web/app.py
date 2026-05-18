#!/usr/bin/env python3
"""
跌倒检测管理后台 - Flask Web UI

启动:
    python web/app.py
    或双击 web/run.bat

功能:
    - 模型选择与训练配置
    - 一键启动训练（后台运行）
    - 实时训练监控（Loss/F1曲线）
    - 测试集评估
    - 结果对比与历史记录
"""
import sys
import os
import re
import json
import time
import signal
import subprocess
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, jsonify, request, Response

# 项目根目录
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = Flask(__name__, template_folder="templates", static_folder="static")

# ─── 全局状态 ─────────────────────────────────────────────────
training_state = {
    "running": False,
    "pid": None,
    "model": None,
    "config": {},
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": [],
    "val_loss": [],
    "val_f1": [],
    "val_precision": [],
    "val_recall": [],
    "val_accuracy": [],
    "lrs": [],
    "best_f1": 0.0,
    "best_epoch": 0,
    "log_lines": [],
    "status": "idle",  # idle, training, completed, error
    "start_time": None,
    "end_time": None,
}
state_lock = threading.Lock()


# ─── 路由 ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/models")
def list_models():
    """可用模型列表"""
    return jsonify({
        "models": [
            {"id": "gait_lstm", "name": "GaitLSTM", "desc": "双向LSTM步态分析模型", "params": "666K"},
            {"id": "gait_transformer", "name": "GaitTransformer", "desc": "Transformer步态分析模型", "params": "~800K"},
        ]
    })


@app.route("/api/checkpoints")
def list_checkpoints():
    """已有检查点列表"""
    ckpt_dir = ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return jsonify({"checkpoints": []})
    
    ckpts = []
    for f in sorted(ckpt_dir.glob("*.pt")):
        stat = f.stat()
        ckpts.append({
            "name": f.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 1),
            "time": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return jsonify({"checkpoints": ckpts})


@app.route("/api/results")
def list_results():
    """已有结果列表"""
    results_dir = ROOT / "results"
    if not results_dir.exists():
        return jsonify({"results": []})
    
    results = []
    for f in sorted(results_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        results.append({
            "name": f.name,
            "metrics": data.get("metrics", {}),
            "model": data.get("model", "unknown"),
            "num_samples": data.get("num_samples", 0),
        })
    return jsonify({"results": results})


@app.route("/api/status")
def get_status():
    """当前训练状态"""
    with state_lock:
        return jsonify(training_state)


@app.route("/api/train", methods=["POST"])
def start_training():
    """启动训练"""
    with state_lock:
        if training_state["running"]:
            return jsonify({"error": "训练正在进行中"}), 400

    config = request.json or {}
    model_name = config.get("model", "gait_lstm")
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 32)
    lr = config.get("lr", 0.001)
    device = config.get("device", "cuda:0")
    data_dir = config.get("data_dir", str(ROOT / "data" / "processed" / "ntu_coco"))

    # 重置状态
    with state_lock:
        training_state.update({
            "running": True,
            "model": model_name,
            "config": config,
            "current_epoch": 0,
            "total_epochs": epochs,
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_accuracy": [],
            "lrs": [],
            "best_f1": 0.0,
            "best_epoch": 0,
            "log_lines": [],
            "status": "training",
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
        })

    # 后台线程运行训练
    thread = threading.Thread(
        target=_run_training,
        args=(model_name, epochs, batch_size, lr, device, data_dir),
        daemon=True,
    )
    thread.start()

    return jsonify({"message": "训练已启动", "model": model_name, "epochs": epochs})


@app.route("/api/stop", methods=["POST"])
def stop_training():
    """停止训练"""
    with state_lock:
        pid = training_state.get("pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        training_state["running"] = False
        training_state["status"] = "stopped"
        training_state["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"message": "训练已停止"})


@app.route("/api/evaluate", methods=["POST"])
def run_evaluate():
    """运行测试集评估"""
    config = request.json or {}
    model_name = config.get("model", "gait_lstm")
    checkpoint = config.get("checkpoint", str(ROOT / "checkpoints" / "best_model.pt"))
    device = config.get("device", "cuda:0")

    cmd = [
        sys.executable, str(ROOT / "scripts" / "evaluate.py"),
        "--model", model_name,
        "--checkpoint", checkpoint,
        "--device", device,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        output = result.stdout + result.stderr
        
        # 加载结果文件
        results_file = ROOT / "results" / "test_results.json"
        metrics = {}
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                metrics = data

        return jsonify({
            "success": True,
            "output": output,
            "results": metrics,
        })
    except subprocess.TimeoutExpired:
        return jsonify({"error": "评估超时"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stream")
def stream():
    """SSE 实时训练数据流"""
    def generate():
        last_len = 0
        last_epoch = -1
        while True:
            with state_lock:
                state = dict(training_state)
            
            if state["status"] in ("completed", "error", "stopped"):
                # 发送最终状态
                yield f"data: {json.dumps(state)}\n\n"
                break
            
            # 只在有新数据时发送
            current_epoch = state["current_epoch"]
            if current_epoch != last_epoch or len(state["log_lines"]) != last_len:
                last_epoch = current_epoch
                last_len = len(state["log_lines"])
                yield f"data: {json.dumps(state)}\n\n"
            
            time.sleep(1)
    
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/history")
def get_history():
    """获取训练历史文件"""
    history_file = ROOT / "checkpoints" / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            return jsonify(json.load(f))
    return jsonify({})


# ─── 后台训练 ─────────────────────────────────────────────────
def _run_training(model_name, epochs, batch_size, lr, device, data_dir):
    """后台执行训练进程"""
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--model", model_name,
        "--dataset", "ntu",
        "--data-dir", data_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--device", device,
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )
        
        with state_lock:
            training_state["pid"] = proc.pid

        # 解析输出
        epoch_pattern = re.compile(
            r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
            r"Train Loss:\s+([\d.]+)\s+\|\s+"
            r"Val Loss:\s+([\d.]+)\s+\|\s+"
            r"Val F1:\s+([\d.]+)\s+\|\s+"
            r"LR:\s+([\d.]+)"
        )
        best_pattern = re.compile(r"新最佳模型.*F1:\s+([\d.]+)")
        completed_pattern = re.compile(r"训练完成.*最佳 F1:\s+([\d.]+)")
        early_stop_pattern = re.compile(r"早停触发")

        for line in proc.stdout:
            line = line.rstrip()
            with state_lock:
                training_state["log_lines"].append(line)
                # 只保留最近 500 行
                if len(training_state["log_lines"]) > 500:
                    training_state["log_lines"] = training_state["log_lines"][-500:]

            # 解析 epoch 数据
            m = epoch_pattern.search(line)
            if m:
                ep, total, tl, vl, vf1, lr_val = m.groups()
                with state_lock:
                    training_state["current_epoch"] = int(ep)
                    training_state["total_epochs"] = int(total)
                    training_state["train_loss"].append(float(tl))
                    training_state["val_loss"].append(float(vl))
                    training_state["val_f1"].append(float(vf1))
                    training_state["lrs"].append(float(lr_val))

            # 解析最佳模型
            m = best_pattern.search(line)
            if m:
                with state_lock:
                    training_state["best_f1"] = float(m.group(1))
                    training_state["best_epoch"] = training_state["current_epoch"]

            # 解析完成
            m = completed_pattern.search(line)
            if m:
                with state_lock:
                    training_state["best_f1"] = float(m.group(1))

            if early_stop_pattern.search(line):
                with state_lock:
                    training_state["log_lines"].append("[系统] 早停触发")

        proc.wait()

        with state_lock:
            training_state["running"] = False
            training_state["pid"] = None
            training_state["status"] = "completed"
            training_state["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        with state_lock:
            training_state["running"] = False
            training_state["status"] = "error"
            training_state["log_lines"].append(f"[ERROR] {str(e)}")
            training_state["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─── 启动 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  跌倒检测管理后台")
    print("  打开浏览器访问: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
