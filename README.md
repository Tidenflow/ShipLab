# ShipGravityLab

船舶重心计算工具。解析 Abaqus INP 格式的有限元网格，通过 CUDA 并行计算各单元质量与形心，汇总得到整船重心坐标，并通过 Python GUI 进行 3D 可视化。

## 架构

```
┌─────────────────────────────────┐     gRPC (50051)     ┌──────────────────────────┐
│  Python 前端 (client_gui.py)    │ ◄──────────────────► │  C++ 后端 (ShipGravityServer) │
│  PySide6 + VTK                  │                      │  CUDA + gRPC             │
└─────────────────────────────────┘                      └──────────────────────────┘
```

- **后端**：读取 INP 文件 → GPU 并行计算面积/体积/重量/形心 → gRPC 返回网格数据和计算结果
- **前端**：接收顶点/索引 → VTK 渲染 3D 网格 → 展示重心坐标和总质量

## 目录结构

```
ShipGravityLab/
├── src/
│   ├── cli/main.cu                     # 本地命令行入口（直接计算，无网络）
│   ├── server/main_server.cpp          # gRPC 服务实现
│   ├── core/GpuOperatorOptimized_v2.cu # CUDA 核函数（面积、重心计算）
│   ├── core/CpuAggregator.cpp          # CPU 汇总（加权平均重心）
│   ├── core/OperatorTags.h             # GPU 数据结构定义
│   └── io/MemoryMappedFileReader.h     # INP 文件内存映射解析器
├── python/
│   ├── client_gui.py                   # PySide6 + VTK 前端 GUI
│   ├── simulation_pb2.py               # Python gRPC 桩代码（已生成）
│   └── simulation_pb2_grpc.py
├── proto/
│   └── simulation.proto                # gRPC 接口定义
├── tests/data/
│   ├── chuanmo.inp                     # 测试船模（完整船体）
│   └── cangduan1-jm.inp                # 测试船模（舱段）
├── build_linux/                        # 编译输出目录
├── scripts/
│   ├── run.sh                          # 本地编译脚本
│   ├── docker_run.sh                   # Docker 构建/运行脚本
│   └── Makefile.linux                  # Linux Makefile
├── Dockerfile
└── CMakeLists.txt
```

## 快速开始（推荐：C/S 模式）

### 1. 启动后端（Linux / WSL2 / Docker）

#### 方式 A：直接运行（需要 CUDA 环境）

```bash
# 安装依赖（Ubuntu/Debian 示例）
sudo apt update
sudo apt install -y build-essential cmake libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc

# 编译
cd ShipGravityLab
rm -rf build_linux && mkdir build_linux && cd build_linux
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89
make -j$(nproc) ShipGravityServer

# 运行后端
./ShipGravityServer
# 输出：[Server] ShipGravityLab 后端已就绪，监听: 0.0.0.0:50051
```

> 如果你的显卡不是 RTX 4060，请修改 `-DCMAKE_CUDA_ARCHITECTURES`：
> - RTX 30 系 → `86`
> - RTX 20 系 / T4 → `75`
> - V100 → `70`
> - A100 / H100 → `80`

#### 方式 B：Docker 运行（推荐，一键部署）

```bash
cd ShipGravityLab
bash scripts/docker_run.sh
```

Docker 镜像会自动编译并启动后端服务，暴露 50051 端口。

### 2. 启动前端（Windows / Linux / macOS）

```bash
# 安装 Python 依赖
pip install PySide6 vtk grpcio grpcio-tools numpy

# 运行前端
cd ShipGravityLab/python
python client_gui.py
```

### 3. 使用流程

1. 在前端输入后端地址 `localhost:50051`（如果后端在远程服务器，填写其 IP），点击**连接后端**
2. 填入 INP 文件路径：
   - 若后端在 **WSL2**，前端支持自动将 Windows 路径（如 `D:\models\ship.inp`）转换为 WSL 路径（`/mnt/d/models/ship.inp`）
   - 若后端在 **Linux 服务器**，直接填写服务器上的绝对路径
3. 点击**加载模型**，等待 3D 网格渲染
4. 设置材料密度（默认 7850 kg/m³），点击**开始计算**
5. 查看重心坐标 (X, Y, Z) 和总质量

---

## 命令行模式（无需网络 / 无 GUI）

如果你只需要快速得到计算结果，可以直接运行本地可执行文件：

```bash
cd build_linux
./ShipGravityLab tests/data/cangduan1-jm.inp
```

---

## 环境依赖

### 后端

| 依赖 | 版本要求 |
|------|---------|
| CUDA | ≥ 11.0（推荐 12.x） |
| CMake | ≥ 3.18 |
| gRPC / Protobuf | 系统包管理器安装 |
| C++ 编译器 | 支持 C++17 |

```bash
# Ubuntu/Debian
sudo apt install -y libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
```

### 前端

```bash
pip install PySide6 vtk grpcio grpcio-tools numpy
```

---

## 支持的 INP 单元类型

| 类型 | 说明 |
|------|------|
| `S3` | 三角形板单元 |
| `S4R` | 四边形板单元 |
| `B31` | 梁单元 |

---

## gRPC 接口

```protobuf
// 获取网格数据（顶点 + 索引）
rpc GetMesh (MeshRequest) returns (MeshResponse);

// 执行重心计算
rpc RunAnalysis (AnalysisRequest) returns (AnalysisResult);
```

- `MeshResponse.vertices`：`[x1,y1,z1, x2,y2,z2, ...]`，`double` 类型
- `MeshResponse.indices`：面片索引，`-1` 作为单元分隔符
- `AnalysisResult`：重心坐标 `(x, y, z)`、总质量、计算耗时

---

## 重新生成 protobuf 桩代码

修改 `proto/simulation.proto` 后需重新生成：

```bash
# C++ 端（由 CMake 自动处理，重新 make 即可）
make -C build_linux ShipGravityServer

# Python 端
python -m grpc_tools.protoc \
    -I./proto \
    --python_out=./python \
    --grpc_python_out=./python \
    ./proto/simulation.proto
```

---

## 常见问题

### Q: 前端提示“连接超时”
- 确认后端已启动且防火墙未拦截 50051 端口
- 若后端在远程服务器，前端地址应填写服务器 IP，而非 `localhost`

### Q: 加载大模型时前端卡顿或崩溃
- 勾选**低负载模式**（跳过网格渲染，仅做计算）
- 或增大**抽稀比例**（如设为 10，只传 10% 的面片用于显示）

### Q: Docker 运行后前端连不上
- 确保启动容器时映射了端口：`-p 50051:50051`
- 检查 `nvidia-container-toolkit` 是否已安装

---

## 许可证

MIT License
