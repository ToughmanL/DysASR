#!/bin/bash

# 日志文件
LOG_FILE="benchmark.log"

# 清理旧的监控日志
rm -f gpu_monitor.log cpu_mem_monitor.log

# 启动 GPU 显存监控（后台运行）
nvidia-smi --id=2 --query-gpu=memory.used --format=csv -l 1 > gpu_monitor.log & GPU_MONITOR_PID=$!

# 启动 CPU 和内存监控（后台运行）
(
    while true; do
        # 监控当前脚本的 CPU 和内存占用（可根据进程名调整）
        ps -p $$ -o %cpu,%mem --no-headers >> cpu_mem_monitor.log
        sleep 0.1  # 更精细的监控间隔
    done
) &
CPU_MEM_MONITOR_PID=$!

# 记录开始时间
START_TIME=$(date +%s.%N)

# 在这里替换为你的实际运行命令（示例：批量处理文件）
CUDA_VISIBLE_DEVICES=2 python evaluation.py --test_data dataset/MSDM/dev.json dataset/MSDM/test.json --model_path=models/MSDM/whisper-base-finetune --result_dir results/MSDM --language Chinese --metric cer --batch_size 1

# 记录结束时间
END_TIME=$(date +%s.%N)

# 计算处理时间（秒）
PROCESS_TIME=$(echo "$END_TIME - $START_TIME" | bc)

# 停止监控进程
kill $GPU_MONITOR_PID $CPU_MEM_MONITOR_PID >/dev/null 2>&1
wait $GPU_MONITOR_PID $CPU_MEM_MONITOR_PID 2>/dev/null

# 提取峰值资源占用
PEAK_GPU=$(grep -o '[0-9]\+' gpu_monitor.log | sort -n | tail -1 || echo "N/A")
PEAK_CPU=$(awk '{print $1}' cpu_mem_monitor.log | sort -n | tail -1 || echo "N/A")
PEAK_MEM=$(awk '{print $2}' cpu_mem_monitor.log | sort -n | tail -1 || echo "N/A")

# 输出结果
echo "===== Benchmark Results ====="
echo "Processing time: ${PROCESS_TIME}s"
echo "Peak GPU memory: ${PEAK_GPU} MB"
echo "Peak CPU usage: ${PEAK_CPU}%"
echo "Peak RAM usage: ${PEAK_MEM}%"

# 保存到日志文件
{
    echo "[$(date)]"
    echo "Command: $0 $@"
    echo "Process time: ${PROCESS_TIME}s"
    echo "Peak GPU memory: ${PEAK_GPU} MB"
    echo "Peak CPU: ${PEAK_CPU}%"
    echo "Peak RAM: ${PEAK_MEM}%"
    echo "------------------------------------"
} >> "$LOG_FILE"

# 清理临时文件
rm -f gpu_monitor.log cpu_mem_monitor.log

echo "Results saved to $LOG_FILE"