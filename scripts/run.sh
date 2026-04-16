#!/bin/bash

# 修改为实际安装的 CUDA 12.1 路径
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=== 开始编译船舶重心计算程序 ==="

# 1. 彻底清理旧编译目录并重新创建
rm -rf build_linux
mkdir -p build_linux
cd build_linux

# 2. 编译
# 显式指定编译器路径为 12.1，并保持 RTX 4060 的架构 89
cmake .. \
    -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89

# 执行构建
if make -j$(nproc); then
    echo "=== 编译成功 ==="
else
    echo "=== 编译失败，请检查错误输出 ==="
    exit 1
fi

cd ..

echo "=== 开始运行计算 ==="
echo "目前运行目录是: $(pwd)"

# 核心：将栈限制从 8MB 改为无限
ulimit -s unlimited

# 使用 gdb 运行
if [ -f "./build_linux/ShipGravityLab" ]; then
     gdb -batch -ex "run" -ex "bt" ./build_linux/ShipGravityLab
else
    echo "错误: 未找到可执行文件"
fi

echo "=== 计算完成 ==="