# 1. 基础镜像：阿里云代理的 CUDA 镜像
FROM registry.cn-hangzhou.aliyuncs.com/google_containers/cuda:12.1.0-devel-ubuntu22.04

# 2. 环境设置
ENV DEBIAN_FRONTEND=noninteractive

# 3. 换 Ubuntu 源为阿里云 + 安装依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    tzdata wget ca-certificates gnupg software-properties-common \
    gcc g++ make git python3-pip cmake \
    libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    pip3 install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# 4. 设置工作目录
WORKDIR /app

# 5. 复制源码
COPY . .

# 6. 编译项目
RUN rm -rf build_linux && mkdir build_linux && cd build_linux && \
    cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. && \
    make -j$(nproc)

# 7. 暴露端口并启动
EXPOSE 50051
CMD ["/app/build_linux/ShipGravityServer"]