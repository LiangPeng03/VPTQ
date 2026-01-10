#!/usr/bin/env python3
import pynvml
import time
import csv
import os
import sys
from datetime import datetime

def initialize_nvml():
    """Initialize NVML (NVIDIA Management Library)."""
    pynvml.nvmlInit()

def get_gpu_info():
    """
    Get GPU utilization and memory usage information for all GPUs.
    
    Returns:
        list: A list of dictionaries containing GPU information for each GPU.
    """
    gpu_count = pynvml.nvmlDeviceGetCount()
    gpu_info_list = []
    
    for i in range(gpu_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 获取GPU名称并处理兼容性问题
            gpu_name_raw = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name_raw, bytes):
                gpu_name = gpu_name_raw.decode('utf-8')
            else:
                gpu_name = str(gpu_name_raw)
            
            gpu_info = {
                'gpu_index': i,
                'gpu_name': gpu_name,
                'gpu_util': utilization.gpu,
                'mem_used': memory.used / 1024**3,  # Convert bytes to GiB
                'mem_total': memory.total / 1024**3,  # Convert bytes to GiB
                'mem_percent': (memory.used / memory.total) * 100 if memory.total > 0 else 0
            }
            gpu_info_list.append(gpu_info)
        except Exception as e:
            print(f"Error getting info for GPU {i}: {e}")
    
    return gpu_info_list

def monitor_gpu(output_file, duration=None, num_gpus=1):
    """
    Monitor GPU usage and save to CSV file.
    
    Args:
        output_file (str): Path to output CSV file
        duration (int, optional): Monitoring duration in seconds. If None, runs indefinitely
        num_gpus (int): Number of GPUs being used for quantization
    """
    initialize_nvml()
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'gpu_index', 'gpu_name', 'gpu_util', 'mem_used', 'mem_total', 'mem_percent']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        start_time = time.time()
        max_memory_per_gpu = [0] * pynvml.nvmlDeviceGetCount()
        total_gpu_hours = 0
        sample_count = 0
        
        try:
            while True:
                # Check if duration limit is set and exceeded
                if duration and (time.time() - start_time) > duration:
                    break
                    
                timestamp = datetime.now().strftime('%Y/%m/%d %H:%M:%S.%f')
                gpu_info_list = get_gpu_info()
                
                for gpu_info in gpu_info_list:
                    gpu_info['timestamp'] = timestamp
                    writer.writerow(gpu_info)
                    
                    # Track maximum memory usage per GPU
                    gpu_index = gpu_info['gpu_index']
                    if gpu_info['mem_used'] > max_memory_per_gpu[gpu_index]:
                        max_memory_per_gpu[gpu_index] = gpu_info['mem_used']
                
                csvfile.flush()  # Ensure data is written to file immediately
                sample_count += 1
                
                # Calculate current stats for display
                if gpu_info_list:
                    total_current_mem = sum(gpu_info['mem_used'] for gpu_info in gpu_info_list)
                    avg_util = sum(gpu_info['gpu_util'] for gpu_info in gpu_info_list) / len(gpu_info_list)
                    
                    # Calculate GPU hours (total time all GPUs have been running)
                    elapsed_time_hours = (time.time() - start_time) / 3600
                    current_gpu_hours = elapsed_time_hours * num_gpus
                    
                    print(f"Time: {elapsed_time_hours:.2f}h, "
                          f"GPU Hours: {current_gpu_hours:.2f}, "
                          f"Total Memory: {total_current_mem:.1f} GiB, "
                          f"Avg Util: {avg_util:.1f}%", end='\r')
                
                time.sleep(5)  # Sample every second
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"\nError occurred: {e}")
        finally:
            # Calculate final stats
            end_time = time.time()
            total_time_seconds = end_time - start_time
            total_time_hours = total_time_seconds / 3600
            total_gpu_hours = total_time_hours * num_gpus
            
            # Print summary
            print(f"\n\n===== GPU Usage Summary =====")
            print(f"Total monitoring time: {total_time_hours:.2f} hours")
            print(f"Number of GPUs monitored: {len(max_memory_per_gpu)}")
            print(f"Total GPU hours: {total_gpu_hours:.2f} GPU hours")
            
            for i, max_mem in enumerate(max_memory_per_gpu):
                if max_mem > 0:
                    print(f"Max memory usage on GPU {i}: {max_mem:.1f} GiB")
            
            print(f"Results saved to: {output_file}")
            
            return total_gpu_hours, max_memory_per_gpu

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_gpu.py <output_csv_file> [num_gpus] [duration_seconds]")
        print("  num_gpus: Number of GPUs being used (default: 1)")
        print("  duration_seconds: Monitoring duration in seconds (default: None, runs indefinitely)")
        sys.exit(1)
    
    output_file = sys.argv[1]
    num_gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    monitor_gpu(output_file, duration, num_gpus)