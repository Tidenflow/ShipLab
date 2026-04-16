#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <cstdio>
#include <unordered_map>

#include <grpcpp/grpcpp.h>
#include "simulation.grpc.pb.h"

// 引入你的核心业务头文件
#include "MemoryMappedFileReader.h"
#include "OperatorTags.h"

using SteadyClock = std::chrono::steady_clock;

// =========================================================
// GPU & CPU 外部算子声明（保持不变）
// =========================================================
extern "C" void initGpuMemoryPool(size_t maxPlateCount, size_t maxBeamCount);
extern "C" void scheduleGpuPlateOps(ShipPlateGpuData* plates, int plateCount);
extern "C" void scheduleGpuBeamOps(ShipBeamGpuData* beams, int beamCount);
extern "C" void cleanupGpuMemoryPool();

extern void calcTotalGravityCenter(const ShipPlateGpuData* plates, int plateCount,
                                   const ShipBeamGpuData* beams, int beamCount,
                                   float& totalWeight, float totalCentroid[3]);

// =========================================================
// 全局缓存：避免重复解析大文件
// =========================================================
struct CachedMesh {
    std::string filePath;
    std::vector<InpNode> nodes;
    std::vector<PlateEntity> plateEntities;
    std::vector<BeamEntity> beamEntities;
    std::vector<ShipPlateGpuData> gpuPlates;
    std::vector<ShipBeamGpuData> gpuBeams;
    bool analysisRun = false;
};

static CachedMesh g_mesh;
static std::mutex g_meshMutex;

// =========================================================
// SimulationServiceImpl - gRPC 服务核心实现
// =========================================================
// 仿真服务实现类，最终类（禁止继承），继承自gRPC自动生成的服务基类
class SimulationServiceImpl final : public sim::SimulationService::Service {

    // --------------------------------------------------
    // 接口 1: GetMesh - 序列化网格并发送至前端
    // 功能：接收客户端的网格请求，读取/缓存模型文件，处理后返回顶点和索引数据
    // --------------------------------------------------
    grpc::Status GetMesh(grpc::ServerContext* context,     // gRPC服务上下文，用于元数据、取消、超时等
                         const sim::MeshRequest* request,  // 客户端请求（包含文件路径、是否跳过数据等）
                         sim::MeshResponse* response) override { // 服务端响应（输出顶点、索引）
        
        // 从客户端请求中获取模型文件路径（const& 避免拷贝，提升效率）
        const std::string& filePath = request->file_path();

        // ==================== 线程安全保护 ====================
        // 对全局网格缓存 g_mesh 加锁
        // gRPC是多线程服务，防止多个线程同时修改/读取全局缓存导致数据竞争、崩溃
        std::lock_guard<std::mutex> lock(g_meshMutex);

        // ==================== 1. 文件解析与全局缓存 ====================
        // 设计目的：避免重复解析大文件，大幅提升服务端性能
        // 如果当前缓存的文件路径 != 请求路径 → 需要重新加载解析
        if (g_mesh.filePath != filePath) {
            // 使用内存映射文件读取器（高效读取大文件，比普通fread快很多）
            MemoryMappedFileReader reader;

            // 尝试打开文件，失败则返回 NOT_FOUND 错误给客户端
            if (!reader.open(filePath)) {
                return grpc::Status(grpc::StatusCode::NOT_FOUND, "无法打开文件");
            }

            // 创建临时缓存对象，用于存储新解析的网格数据
            CachedMesh newMesh;
            newMesh.filePath = filePath; // 记录文件路径，用于后续缓存判断

            // 解析文件：读取节点、板单元、梁单元数据
            // 解析失败 → 返回 INTERNAL 内部错误
            if (!reader.parse(newMesh.nodes, newMesh.plateEntities, newMesh.beamEntities)) {
                return grpc::Status(grpc::StatusCode::INTERNAL, "文件解析失败");
            }

            // 将新解析的数据移动赋值给全局缓存（std::move 零拷贝，高性能）
            // 避免拷贝大量顶点/面片数据，节省内存与时间
            g_mesh = std::move(newMesh);
        }

        // ==================== 可选：跳过网格数据传输 ====================
        // 客户端可设置 skip_mesh_data=true，仅检查文件是否存在/可解析，不返回模型数据
        if (request->skip_mesh_data()) 
            return grpc::Status::OK;

        // ==================== 2. 建立 ID → 连续索引映射（核心：解决渲染碎裂） ====================
        // 原理：
        // 模型文件中的节点ID通常是不连续的（如 1001, 2005...）
        // 渲染引擎必须使用 0,1,2,3... 连续索引
        // 建立映射表，把原始ID转为连续索引，防止渲染碎裂、黑面、崩溃
        std::unordered_map<int, int> idMap;
        
        // 预先分配哈希表空间，避免运行时频繁扩容，提升性能
        idMap.reserve(g_mesh.nodes.size());

        // 预先为顶点数据分配内存：每个节点3个float（x,y,z）
        // Reserve 减少protobuf repeated字段多次扩容开销
        response->mutable_vertices()->Reserve(static_cast<int>(g_mesh.nodes.size() * 3));

        // 遍历所有节点，构建ID映射表，并填充顶点数据
        for (int i = 0; i < (int)g_mesh.nodes.size(); ++i) {
            const auto& n = g_mesh.nodes[i]; // 当前节点（引用，不拷贝）
            
            // 核心映射：原始节点ID → 连续数组索引（0,1,2...）
            idMap[n.id] = i;

            // 将节点坐标 x/y/z 依次加入protobuf响应
            // 顶点数据格式：扁平数组 [x0,y0,z0, x1,y1,z1, x2,y2,z2 ...]
            response->add_vertices(n.x);
            response->add_vertices(n.y);
            response->add_vertices(n.z);
        }

        // ==================== 调试：输出模型坐标范围 ====================
        // 用于服务端排查问题：模型坐标是否异常、是否太小/太大、是否看不见
        if (!g_mesh.nodes.empty()) {
            // 初始化极值为第一个节点坐标
            float x_min = g_mesh.nodes[0].x, x_max = g_mesh.nodes[0].x;
            float y_min = g_mesh.nodes[0].y, y_max = g_mesh.nodes[0].y;
            float z_min = g_mesh.nodes[0].z, z_max = g_mesh.nodes[0].z;

            // 遍历所有节点，计算真实坐标范围
            for (const auto& n : g_mesh.nodes) {
                x_min = std::min(x_min, n.x); x_max = std::max(x_max, n.x);
                y_min = std::min(y_min, n.y); y_max = std::max(y_max, n.y);
                z_min = std::min(z_min, n.z); z_max = std::max(z_max, n.z);
            }

            // 控制台打印坐标范围，方便调试
            std::cout << "[Server] 模型坐标范围: X=[" << x_min << ", " << x_max
                      << "], Y=[" << y_min << ", " << y_max
                      << "], Z=[" << z_min << ", " << z_max << "]" << std::endl;
        }

        // ==================== 3. 填充面片索引 + 四边形三角化 ====================
        // 获取响应中的索引数组指针
        auto* resp_indices = response->mutable_indices();

        // 遍历所有板单元（面片/面单元）
        for (const auto& p : g_mesh.plateEntities) {
            // 只处理有效面片：至少3个节点才能构成面
            if (p.nodeCount >= 3) {
                try {
                // ==============================================
                // 【新版本】不拆三角形！
                // 3 节点 → 直接传 3 个索引
                // 4 节点 → 直接传 4 个索引
                // ==============================================
                if (p.nodeCount == 3) {
                    // 三角形：直接添加 3 个索引
                    resp_indices->Add(idMap.at(p.nodeIds[0]));
                    resp_indices->Add(idMap.at(p.nodeIds[1]));
                    resp_indices->Add(idMap.at(p.nodeIds[2]));
                    resp_indices->Add(-1); // 结束标记
                }
                else if (p.nodeCount == 4) {
                    // 四边形：直接添加 4 个索引！不拆分！
                    resp_indices->Add(idMap.at(p.nodeIds[0]));
                    resp_indices->Add(idMap.at(p.nodeIds[1]));
                    resp_indices->Add(idMap.at(p.nodeIds[2]));
                    resp_indices->Add(idMap.at(p.nodeIds[3]));
                    resp_indices->Add(-1); // 结束标记
                }
            }  catch (...) { 
                    // 异常捕获：idMap.at() 找不到ID时会抛异常
                    // 跳过错误面片，保证服务不崩溃，继续处理其他面片
                    continue; 
                }
            }
        }

        // 所有数据处理完成，返回成功状态给客户端
        return grpc::Status::OK;
    }

    // --------------------------------------------------
    // 接口 2: RunAnalysis - 执行 CUDA 计算并汇总
    // --------------------------------------------------
    grpc::Status RunAnalysis(grpc::ServerContext* context,
                             const sim::AnalysisRequest* request,
                             sim::AnalysisResult* response) override {

        std::cout << "[Server][RunAnalysis] 收到计算请求, density=" << request->density() << std::endl;

        std::lock_guard<std::mutex> lock(g_meshMutex);
        if (g_mesh.filePath.empty()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "尚未加载模型");
        }

        auto t0 = SteadyClock::now();

        // 如果密度改变，标记需要重新计算
        if (request->density() > 0.0) {
            float d = (float)request->density();
            for (auto& p : g_mesh.gpuPlates) p.density = d;
            for (auto& b : g_mesh.gpuBeams)  b.density = d;
            g_mesh.analysisRun = false; 
        }

        if (!g_mesh.analysisRun) {
            initGpuMemoryPool(g_mesh.gpuPlates.size() * 2, g_mesh.gpuBeams.size() * 2);
            if (!g_mesh.gpuPlates.empty()) 
                scheduleGpuPlateOps(g_mesh.gpuPlates.data(), (int)g_mesh.gpuPlates.size());
            if (!g_mesh.gpuBeams.empty()) 
                scheduleGpuBeamOps(g_mesh.gpuBeams.data(), (int)g_mesh.gpuBeams.size());
            g_mesh.analysisRun = true;
        }

        float totalWeight = 0.0f;
        float totalCentroid[3] = {0.0f, 0.0f, 0.0f};
        calcTotalGravityCenter(g_mesh.gpuPlates.data(), (int)g_mesh.gpuPlates.size(),
                               g_mesh.gpuBeams.data(),  (int)g_mesh.gpuBeams.size(),
                               totalWeight, totalCentroid);

        auto t1 = SteadyClock::now();
        double execMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        response->set_x(totalCentroid[0]);
        response->set_y(totalCentroid[1]);
        response->set_z(totalCentroid[2]);
        response->set_total_mass(totalWeight);
        response->set_execution_time_ms(execMs);

        return grpc::Status::OK;
    }
};

// =========================================================
// Main 启动函数
// =========================================================
int main(int argc, char* argv[]) {
    std::string serverAddr = "0.0.0.0:50051";
    SimulationServiceImpl service;

    grpc::ServerBuilder builder;
    builder.AddListeningPort(serverAddr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    // 设置消息限制为 512MB，防止大网格传输失败
    builder.SetMaxReceiveMessageSize(512 * 1024 * 1024);
    builder.SetMaxSendMessageSize(512 * 1024 * 1024);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "[Server] ShipGravityLab 后端已就绪，监听: " << serverAddr << std::endl;

    server->Wait();
    cleanupGpuMemoryPool();
    return 0;
}