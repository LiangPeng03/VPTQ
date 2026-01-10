#!/bin/bash

# GPU数量（与--num_gpus参数保持一致）
NUM_GPUS=2

# 创建输出目录（基于当前时间）
OUTPUT_DIR="/home/pengliang/Desktop/test_output/Llama-2-7_hessian/$(date +%Y-%m-%d-%H-%M-%S)"
mkdir -p "$OUTPUT_DIR"

# GPU监控日志文件
GPU_LOG="$OUTPUT_DIR/gpu_monitoring.csv"

# 启动GPU监控进程
echo "Starting GPU monitoring for $NUM_GPUS GPUs..."
python3 monitor_gpu.py "$GPU_LOG" $NUM_GPUS & 
MONITOR_PID=$!

# 等待监控启动
sleep 2

# 运行量化过程
python hessian_collector.py \
    --base_model /home/pengliang/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
    --save_path /home/pengliang/Desktop/local_models/llama2_7b/hessians \
    --gpus "1" \

# 量化完成后停止监控
kill $MONITOR_PID

echo "Quantization completed. GPU usage data saved to: $GPU_LOG"

# 分析GPU使用情况
echo "Analyzing GPU usage..."
python -c "
import csv
import sys

try:
    with open('$GPU_LOG', 'r') as f:
        reader = csv.DictReader(f)
        records = list(reader)
        
    if len(records) > 0:
        # Group records by GPU index
        gpu_records = {}
        for record in records:
            gpu_index = record['gpu_index']
            if gpu_index not in gpu_records:
                gpu_records[gpu_index] = []
            gpu_records[gpu_index].append(record)
        
        print('===== GPU Usage Analysis =====')
        total_max_mem = 0
        for gpu_index, records in gpu_records.items():
            if len(records) > 0:
                max_mem = max(float(r['mem_used']) for r in records)
                avg_mem = sum(float(r['mem_used']) for r in records) / len(records)
                max_util = max(float(r['gpu_util']) for r in records)
                avg_util = sum(float(r['gpu_util']) for r in records) / len(records)
                
                print(f'GPU {gpu_index}:')
                print(f'  Max Memory Usage: {max_mem:.1f} GiB')
                print(f'  Avg Memory Usage: {avg_mem:.1f} GiB')
                print(f'  Max Utilization: {max_util:.1f}%')
                print(f'  Avg Utilization: {avg_util:.1f}%')
                
                total_max_mem += max_mem
        
        # Calculate GPU hours
        sample_interval = 1  # 1 second per sample
        total_time_seconds = len(records) * sample_interval
        total_time_hours = total_time_seconds / 3600
        total_gpu_hours = total_time_hours * len(gpu_records)
        
        print(f'Total Max Memory Usage: {total_max_mem:.1f} GiB')
        print(f'Total Monitoring Time: {total_time_hours:.2f} hours')
        print(f'Number of GPUs: {len(gpu_records)}')
        print(f'Total GPU Hours: {total_gpu_hours:.2f} GPU hours')
    else:
        print('No GPU usage data found.')
except Exception as e:
    print(f'Error analyzing GPU usage: {e}')
    import traceback
    traceback.print_exc()
"