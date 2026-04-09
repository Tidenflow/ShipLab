# 1. 基础镜像：改用阿里云托管的镜像源，解决 Docker Hub 429 频率限制问题
# 如果此源失效，可尝试 registry.cn-hangzhou.aliyuncs.com/google_containers/cuda:12.1.0-devel-ubuntu22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 2. 环境设置
ENV DEBIAN_FRONTEND=noninteractive

# 3. 安装工具
# 将 cmake 移至 apt-get 安装，比 pip 安装更稳定，且速度更快
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    wget \
    ca-certificates \
    gnupg \
    software-properties-common \
    gcc \
    g++ \
    make \
    git \
    python3-pip \
    cmake && \
    # 设置时区
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    # 仅升级 pip，使用阿里源提高稳定性
    pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip && \
    # 清理缓存减少镜像体积
    rm -rf /var/lib/apt/lists/*

# 4. 设置容器内的工作目录
WORKDIR /app

# 5. 复制源码
COPY . .

# 6. 编译项目
# 强制指定架构为 89 (对应你的 RTX 4060)
RUN rm -rf build_linux && mkdir build_linux && cd build_linux && \
    cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. && \
    make -j$(nproc)

# 7. 启动命令
# 确保路径与编译输出一致
CMD ["/app/build_linux/ShipGravityLab", "testconfigs/cangduan1-jm.inp"]