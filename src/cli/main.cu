#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <unistd.h>
#include <limits.h>

#include "MemoryMappedFileReader.h"
#include "OperatorTags.h"

using SteadyClock = std::chrono::steady_clock;

// GPU 外部函数声明
extern "C" void initGpuMemoryPool(size_t maxPlateCount, size_t maxBeamCount);
extern "C" void scheduleGpuPlateOps(ShipPlateGpuData* plates, int plateCount);
extern "C" void scheduleGpuBeamOps(ShipBeamGpuData* beams, int beamCount);

// CPU 汇总函数声明 (实现在其他 cpp 中)
extern void calcTotalGravityCenter(const ShipPlateGpuData* plates, int plateCount,
                                   const ShipBeamGpuData* beams, int beamCount,
                                   float& totalWeight, float totalCentroid[3]);

int main(int argc, char* argv[]) {
    auto programStart = SteadyClock::now();

    // 0. 环境诊断
    char cwd[PATH_MAX];
    std::cout << "------------------------------------------------------------" << std::endl;
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "[系统信息] 当前工作目录: " << cwd << std::endl;
    }

    std::string inpPath = "testconfigs/cangduan1-jm.inp"; 
    if (argc > 1) inpPath = argv[1];
    std::cout << "[系统信息] 目标文件路径: " << inpPath << std::endl;

    // 1. I/O 计时
    auto ioStart = SteadyClock::now();
    MemoryMappedFileReader mmapReader;
    if (!mmapReader.open(inpPath)) {
        std::cerr << "!!! 错误: 无法打开文件: " << inpPath << std::endl;
        return 1;
    }
    auto ioEnd = SteadyClock::now();

    // 2. 解析计时
    std::vector<InpNode> nodes;
    std::vector<PlateEntity> plateEntities;
    std::vector<BeamEntity> beamEntities;

    auto parseStart = SteadyClock::now();
    bool parseSuccess = mmapReader.parse(nodes, plateEntities, beamEntities);
    auto parseEnd = SteadyClock::now();
    if (!parseSuccess) return 1;

    // 3. GPU 预热与池分配计时
    auto gpuPrepStart = SteadyClock::now();
    auto& plates = mmapReader.gpuPlates;
    auto& beams  = mmapReader.gpuBeams;
    
    if (!plates.empty() || !beams.empty()) {
        // 这里会调用 kernel.cu 里的函数，内部已有细分计时打印
        initGpuMemoryPool(plates.size() * 2, beams.size() * 2);
    }
    auto gpuPrepEnd = SteadyClock::now();

    // 4. GPU 算子执行计时
    auto gpuExecStart = SteadyClock::now();
    if (!plates.empty()) scheduleGpuPlateOps(plates.data(), static_cast<int>(plates.size()));
    if (!beams.empty()) scheduleGpuBeamOps(beams.data(), static_cast<int>(beams.size()));
    auto gpuExecEnd = SteadyClock::now();

    // 5. CPU 汇总计时
    auto aggStart = SteadyClock::now();
    float totalWeight = 0.0f;
    float totalCentroid[3] = {0.0f, 0.0f, 0.0f};
    calcTotalGravityCenter(plates.data(), static_cast<int>(plates.size()),
                           beams.data(), static_cast<int>(beams.size()),
                           totalWeight, totalCentroid);
    auto aggEnd = SteadyClock::now();

    // 性能汇总
    auto get_ms = [](auto start, auto end) { 
        return std::chrono::duration<double, std::milli>(end - start).count(); 
    };

    std::cout << "\n>>>>>> 性能分析报告 (单位: ms) <<<<<<" << std::endl;
    std::cout << "1. 文件映射 (I/O):     " << get_ms(ioStart, ioEnd) << " ms" << std::endl;
    std::cout << "2. 文本解析 (Parse):   " << get_ms(parseStart, parseEnd) << " ms" << std::endl;
    std::cout << "3. GPU 预热/池分配:     " << get_ms(gpuPrepStart, gpuPrepEnd) << " ms" << std::endl;
    std::cout << "4. GPU 算子执行:       " << get_ms(gpuExecStart, gpuExecEnd) << " ms" << std::endl;
    std::cout << "5. CPU 结果汇总:       " << get_ms(aggStart, aggEnd) << " ms" << std::endl;
    
    double accounted = get_ms(ioStart, ioEnd) + get_ms(parseStart, parseEnd) + 
                       get_ms(gpuPrepStart, gpuPrepEnd) + get_ms(gpuExecStart, gpuExecEnd) + 
                       get_ms(aggStart, aggEnd);
    double total = get_ms(programStart, aggEnd);

    std::cout << "------------------------------------" << std::endl;
    std::cout << "已统计耗时总和:        " << accounted << " ms" << std::endl;
    std::cout << "程序实际总运行耗时:    " << total << " ms" << std::endl;

    std::cout << "\n>>>>>> 计算结果 <<<<<<" << std::endl;
    std::cout << "全船总重量: " << totalWeight << " kg" << std::endl;
    std::cout << "重心坐标:   (" << totalCentroid[0] << ", " << totalCentroid[1] << ", " << totalCentroid[2] << ") m" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    return 0;
}