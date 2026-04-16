# 基础镜像：用官方 NVidia 镜像（ACR 构建服务可以访问）
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 换 Ubuntu 软件源为阿里云（解决 apt 下载慢）
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

WORKDIR /app
COPY . .

RUN rm -rf build_linux && mkdir build_linux && cd build_linux && \
    cmake -DCMAKE_CUDA_ARCHITECTURES=89 .. && \
    make -j$(nproc)

EXPOSE 50051
CMD ["/app/build_linux/ShipGravityServer"]