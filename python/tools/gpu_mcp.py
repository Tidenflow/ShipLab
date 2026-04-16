import subprocess
import os
import json
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP
mcp = FastMCP("ShipGravityLab_Optimizer")

# --- 工具 1：执行 NCU 性能采样 ---
@mcp.tool()
def run_ncu_profile(executable_path: str = "./build_linux/ShipGravityLab"):
    """
    运行 Nsight Compute 采集性能数据并导出报告。
    """
    report_file = "gpu_report"
    # 使用 sudo 确保有权限访问硬件计数器
    # 注意：建议在终端先跑一次 sudo 确保密码已缓存，或者配置 sudoers 免密
    command = [
        "sudo", "ncu", 
        "--set", "full", 
        "--export", report_file, 
        "--force-overwrite", 
        executable_path
    ]
    
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        return f"✅ 性能分析完成。报告已生成：{report_file}.ncu-rep\n控制台日志：{process.stdout[-200:]}"
    except subprocess.CalledProcessError as e:
        return f"❌ NCU 运行失败：{e.stderr}"

# --- 工具 2：解析核心性能指标（数据脱水） ---
@mcp.tool()
def get_gpu_metrics(report_path: str = "gpu_report.ncu-rep"):
    """
    读取 ncu 报告并提取 Speed of Light (SOL) 核心指标。
    """
    # 只提取最关键的两个指标：显存带宽利用率和计算单元利用率
    # 指标名称：gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (SOL FB)
    #          gpu__compute_status.avg.pct_of_peak_sustained_elapsed (SOL Compute)
    command = [
        "ncu", "--import", report_path, 
        "--metrics", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_status.avg.pct_of_peak_sustained_elapsed"
    ]
    
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True)
        # 简单过滤，只保留包含指标结果的行
        lines = [l.strip() for l in process.stdout.split('\n') if "%" in l or "Metric Name" in l]
        return "\n".join(lines) if lines else "未能提取到核心指标，请确认报告文件是否正确。"
    except Exception as e:
        return f"❌ 指标解析失败：{str(e)}"

# --- 工具 3：编译项目 ---
@mcp.tool()
def build_project(build_dir: str = "build_linux"):
    """
    调用 CMake 编译当前的 CUDA 项目。
    """
    try:
        # 假设你已经配置过 cmake，这里直接 build
        process = subprocess.run(["cmake", "--build", build_dir], capture_output=True, text=True, check=True)
        return f"✅ 编译成功：\n{process.stdout[-300:]}"
    except subprocess.CalledProcessError as e:
        return f"❌ 编译失败：\n{e.stderr}"

if __name__ == "__main__":
    mcp.run()