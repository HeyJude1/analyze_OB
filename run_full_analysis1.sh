#!/bin/bash
# OpenBLAS 真正的Supervisor模式分析 - 后台运行脚本

# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh

# 激活虚拟环境
conda activate agent

# 切换到工作目录
cd /home/dgc/mjs/project/analyze_OB

# 生成日志文件名
LOG_FILE="supervisor_analysis_log_$(date +%Y%m%d_%H%M%S).txt"

# 自动选择全部分析模式（选项2）- 真正的Supervisor智能决策
echo "2" | nohup python agent_work1.py > "$LOG_FILE" 2>&1 &

# 获取进程ID
PID=$!

echo "=========================================="
echo "🧠 真正的Supervisor模式分析已启动！"
echo "=========================================="
echo "进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "虚拟环境: agent"
echo "模式: 智能Supervisor决策"
echo ""
echo "🧠 Supervisor特性："
echo "  ✅ LLM智能决策每一步行动"
echo "  ✅ 自适应错误处理和重试"
echo "  ✅ 动态资源管理和优化"
echo "  ✅ 质量评估和性能监控"
echo ""
echo "📊 监控命令："
echo "  查看实时日志: tail -f $LOG_FILE"
echo "  查看进程状态: ps aux | grep agent_work1.py"
echo "  停止分析: kill $PID"
echo ""
echo "📁 结果将保存在: results/*_supervisor/ 目录"
echo "📋 Supervisor决策日志: results/*_supervisor/supervisor_logs/"
echo "=========================================="
