#!/bin/bash

# --- 配置区 ---
IMAGE_NAME="ship-gravity-lab"
TAG="v2.0"
CONTAINER_NAME="ship_gravity_run"

# 1. 预清理：防止宿主机的编译残留干扰 Docker
echo "正在清理本地残留文件..."
rm -rf build_linux

# 2. 构建镜像
echo "开始构建 Docker 镜像 (Version: ${TAG})..."
# 使用 --no-cache 确保 GLIBC 环境彻底刷新
docker build --no-cache -t ${IMAGE_NAME}:${TAG} .

# 3. 检查并清理旧容器
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "清理旧容器..."
    docker rm -f ${CONTAINER_NAME}
fi

# 4. 运行容器
# 确保你的环境已安装 nvidia-container-toolkit
echo "启动容器进行 CUDA 加速计算..."
docker run --gpus all \
    -p 50051:50051 \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME}:${TAG}