#!/bin/bash
# OpenBLAS 全部分析 - 后台运行脚本

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh

# 激活虚拟环境
conda activate agent

# 切换到工作目录
cd /home/dgc/mjs/project/analyze_OB

# 生成日志文件名
LOG_FILE="analysis_log_$(date +%Y%m%d_%H%M%S).txt"

# 自动选择全部分析模式（选项2）
echo "2" | nohup python agent_work.py > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!

echo "=========================================="
echo "✅ OpenBLAS 全部分析已启动！"
echo "=========================================="
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "虚拟环境: agent"
echo ""
echo "📊 监控命令："
echo "  查看实时日志: tail -f $LOG_FILE"
echo "  查看进程状态: ps aux | grep agent_work.py"
echo "  停止分析: kill $PID"
echo ""
echo "📁 结果将保存在: results/ 目录"
echo "=========================================="

