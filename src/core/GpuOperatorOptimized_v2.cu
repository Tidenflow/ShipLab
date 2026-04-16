#include "OperatorTags.h"
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <chrono>

// 使用稳健的单调时钟
using SteadyClock = std::chrono::steady_clock;

// 计时辅助工具函数
inline double get_ms(SteadyClock::time_point start, SteadyClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// =========================================================
// 1. 核函数
// =========================================================

__global__ void calcPlatePropsKernel_ultra_tight(ShipPlateGpuData* plates, int plateCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= plateCount) return;

    ShipPlateGpuData* plate = &plates[idx];
    float area = 0, cx = 0, cy = 0, cz = 0;

    if (plate->nodeCount == 3) {
        float* a = plate->p1; float* b = plate->p2; float* c = plate->p3;
        float crx = (b[1]-a[1])*(c[2]-a[2]) - (b[2]-a[2])*(c[1]-a[1]);
        float cry = (b[2]-a[2])*(c[0]-a[0]) - (b[0]-a[0])*(c[2]-a[2]);
        float crz = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
        area = 0.5f * sqrtf(crx*crx + cry*cry + crz*crz);
        cx = (a[0] + b[0] + c[0]) * 0.3333333f;
        cy = (a[1] + b[1] + c[1]) * 0.3333333f;
        cz = (a[2] + b[2] + c[2]) * 0.3333333f;
    } 
    else if (plate->nodeCount == 4) {
        float* p1 = plate->p1; float* p2 = plate->p2; float* p3 = plate->p3; float* p4 = plate->p4;
        float cr1x = (p2[1]-p1[1])*(p3[2]-p1[2]) - (p2[2]-p1[2])*(p3[1]-p1[1]);
        float cr1y = (p2[2]-p1[2])*(p3[0]-p1[0]) - (p2[0]-p1[0])*(p3[2]-p1[2]);
        float cr1z = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]);
        float a1 = 0.5f * sqrtf(cr1x*cr1x + cr1y*cr1y + cr1z*cr1z);
        float cr2x = (p3[1]-p1[1])*(p4[2]-p1[2]) - (p3[2]-p1[2])*(p4[1]-p1[1]);
        float cr2y = (p3[2]-p1[2])*(p4[0]-p1[0]) - (p3[0]-p1[0])*(p4[2]-p1[2]);
        float cr2z = (p3[0]-p1[0])*(p4[1]-p1[1]) - (p3[1]-p1[1])*(p4[0]-p1[0]);
        float a2 = 0.5f * sqrtf(cr2x*cr2x + cr2y*cr2y + cr2z*cr2z);
        area = a1 + a2;
        if (area > 1e-12f) {
            float inv = 0.3333333f / area;
            cx = (((p1[0]+p2[0]+p3[0])*a1) + ((p1[0]+p3[0]+p4[0])*a2)) * inv;
            cy = (((p1[1]+p2[1]+p3[1])*a1) + ((p1[1]+p3[1]+p4[1])*a2)) * inv;
            cz = (((p1[2]+p2[2]+p3[2])*a1) + ((p1[2]+p3[2]+p4[2])*a2)) * inv;
        } else {
            cx = (p1[0]+p2[0]+p3[0]+p4[0])*0.25f;
            cy = (p1[1]+p2[1]+p3[1]+p4[1])*0.25f;
            cz = (p1[2]+p2[2]+p3[2]+p4[2])*0.25f;
        }
    }
    plate->area = area;
    plate->centroid[0] = cx; plate->centroid[1] = cy; plate->centroid[2] = cz;
    float vol = area * plate->thickness;
    plate->volume = vol;
    plate->weight = vol * plate->density;
}

__global__ void calcBeamPropsKernel(ShipBeamGpuData* beams, int beamCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= beamCount) return;
    ShipBeamGpuData* b = &beams[idx];
    float dx = b->p2[0] - b->p1[0];
    float dy = b->p2[1] - b->p1[1];
    float dz = b->p2[2] - b->p1[2];
    float len = sqrtf(dx*dx + dy*dy + dz*dz);
    b->length = len;
    b->centroid[0] = (b->p1[0] + b->p2[0]) * 0.5f;
    b->centroid[1] = (b->p1[1] + b->p2[1]) * 0.5f;
    b->centroid[2] = (b->p1[2] + b->p2[2]) * 0.5f;
    float vol = len * b->sectionArea;
    b->volume = vol;
    b->weight = vol * b->density;
}

// =========================================================
// 2. GPU 法向量计算核函数
// =========================================================

__global__ void calcTriangleNormalsKernel(
    const float* vertices,     // [n_verts * 3]
    const int* indices,        // [n_tris * 3]
    float* normals,            // [n_verts * 3] 输出
    int n_tris,
    int n_verts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tris) return;

    int i0 = indices[idx * 3];
    int i1 = indices[idx * 3 + 1];
    int i2 = indices[idx * 3 + 2];

    const float* p0 = vertices + i0 * 3;
    const float* p1 = vertices + i1 * 3;
    const float* p2 = vertices + i2 * 3;

    float ax = p1[0] - p0[0];
    float ay = p1[1] - p0[1];
    float az = p1[2] - p0[2];

    float bx = p2[0] - p0[0];
    float by = p2[1] - p0[1];
    float bz = p2[2] - p0[2];

    float nx = ay * bz - az * by;
    float ny = az * bx - ax * bz;
    float nz = ax * by - ay * bx;

    float len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (len > 1e-12f) {
        nx /= len; ny /= len; nz /= len;
    }

    // 原子累加到三个顶点的法向量（简单平均）
    atomicAdd(normals + i0 * 3, nx);
    atomicAdd(normals + i0 * 3 + 1, ny);
    atomicAdd(normals + i0 * 3 + 2, nz);

    atomicAdd(normals + i1 * 3, nx);
    atomicAdd(normals + i1 * 3 + 1, ny);
    atomicAdd(normals + i1 * 3 + 2, nz);

    atomicAdd(normals + i2 * 3, nx);
    atomicAdd(normals + i2 * 3 + 1, ny);
    atomicAdd(normals + i2 * 3 + 2, nz);
}

__global__ void normalizeNormalsKernel(float* normals, int n_verts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_verts) return;

    float* n = normals + idx * 3;
    float len = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    if (len > 1e-12f) {
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
    }
}

// =========================================================
// 3. 内存池
// =========================================================

struct GpuMemoryPool {
    ShipPlateGpuData* d_plates = nullptr;
    ShipBeamGpuData* d_beams = nullptr;
    size_t maxPlates = 0; size_t maxBeams = 0;
    bool initialized = false;

    bool initialize(size_t maxP, size_t maxB) {
        if (initialized) cleanup();
        
        // 预热驱动
        cudaFree(0); 

        printf("[GPU] 开始显存池预分配...\n");
        auto s1 = SteadyClock::now();
        cudaError_t err1 = cudaMalloc(&d_plates, maxP * sizeof(ShipPlateGpuData));
        auto s2 = SteadyClock::now();
        printf("[池分配计时] Plate 显存开辟: %.4f ms\n", get_ms(s1, s2));

        if (err1 != cudaSuccess) return false;

        cudaError_t err2 = cudaMalloc(&d_beams, maxB * sizeof(ShipBeamGpuData));
        auto s3 = SteadyClock::now();
        printf("[池分配计时] Beam 显存开辟: %.4f ms\n", get_ms(s2, s3));

        if (err2 != cudaSuccess) { cudaFree(d_plates); return false; }

        maxPlates = maxP; maxBeams = maxB;
        initialized = true;
        return true;
    }

    void cleanup() {
        if (d_plates) cudaFree(d_plates);
        if (d_beams) cudaFree(d_beams);
        d_plates = nullptr; d_beams = nullptr;
        initialized = false;
    }
};

static GpuMemoryPool g_memoryPool;

// =========================================================
// 4. 详细计时调度接口
// =========================================================

extern "C" void initGpuMemoryPool(size_t maxP, size_t maxB) {
    auto s = SteadyClock::now();
    if (!g_memoryPool.initialize(maxP, maxB)) {
        printf("[GPU错误] 显存池初始化失败!\n");
    } else {
        printf("[GPU] 显存池就绪: 容量 %zu 板 / %zu 梁\n", maxP, maxB);
    }
    auto e = SteadyClock::now();
    printf("[GPU计时] 内存池总初始化时间: %.4f ms\n", get_ms(s, e));
}

extern "C" void scheduleGpuPlateOps(ShipPlateGpuData* plates, int plateCount) {
    if (!plates || plateCount <= 0) return;
    auto s_total = SteadyClock::now();

    // 1. 地址分配/获取
    auto s_alloc = SteadyClock::now();
    ShipPlateGpuData* d_p = (g_memoryPool.initialized && (size_t)plateCount <= g_memoryPool.maxPlates) 
                            ? g_memoryPool.d_plates : nullptr;
    bool tempAlloc = false;
    if (!d_p) { 
        cudaMalloc(&d_p, plateCount * sizeof(ShipPlateGpuData)); 
        tempAlloc = true; 
    }
    auto e_alloc = SteadyClock::now();

    // 2. 数据上传 (H2D)
    auto s_h2d = SteadyClock::now();
    cudaMemcpy(d_p, plates, plateCount * sizeof(ShipPlateGpuData), cudaMemcpyHostToDevice);
    auto e_h2d = SteadyClock::now();

    // 3. 执行计算 (Kernel)
    auto s_kernel = SteadyClock::now();
    int blockSize = 256;
    int gridSize = (plateCount + blockSize - 1) / blockSize;
    calcPlatePropsKernel_ultra_tight<<<gridSize, blockSize>>>(d_p, plateCount);
    cudaDeviceSynchronize(); // 必须同步，否则计时不准
    auto e_kernel = SteadyClock::now();

    // 4. 数据回传 (D2H)
    auto s_d2h = SteadyClock::now();
    cudaMemcpy(plates, d_p, plateCount * sizeof(ShipPlateGpuData), cudaMemcpyDeviceToHost);
    auto e_d2h = SteadyClock::now();

    if (tempAlloc) cudaFree(d_p);

    auto e_total = SteadyClock::now();

    // 打印明细
    printf("--------------------------------------------------\n");
    printf("[板单元 GPU 明细计时 - 数量: %d]\n", plateCount);
    printf("  > 显存获取 (Alloc): %.4f ms %s\n", get_ms(s_alloc, e_alloc), tempAlloc ? "(动态分配)" : "(池复用)");
    printf("  > 数据上传 (H2D):   %.4f ms\n", get_ms(s_h2d, e_h2d));
    printf("  > 核函数计算(Exec): %.4f ms\n", get_ms(s_kernel, e_kernel));
    printf("  > 数据回传 (D2H):   %.4f ms\n", get_ms(s_d2h, e_d2h));
    printf("  > 板单元调度总计:   %.4f ms\n", get_ms(s_total, e_total));
    printf("--------------------------------------------------\n");
}

extern "C" void scheduleGpuBeamOps(ShipBeamGpuData* beams, int beamCount) {
    if (!beams || beamCount <= 0) return;
    auto s_total = SteadyClock::now();

    // 1. 地址获取
    auto s_alloc = SteadyClock::now();
    ShipBeamGpuData* d_b = (g_memoryPool.initialized && (size_t)beamCount <= g_memoryPool.maxBeams) 
                            ? g_memoryPool.d_beams : nullptr;
    bool tempAlloc = false;
    if (!d_b) { cudaMalloc(&d_b, beamCount * sizeof(ShipBeamGpuData)); tempAlloc = true; }
    auto e_alloc = SteadyClock::now();

    // 2. H2D
    auto s_h2d = SteadyClock::now();
    cudaMemcpy(d_b, beams, beamCount * sizeof(ShipBeamGpuData), cudaMemcpyHostToDevice);
    auto e_h2d = SteadyClock::now();

    // 3. Kernel
    auto s_kernel = SteadyClock::now();
    int blockSize = 256;
    int gridSize = (beamCount + blockSize - 1) / blockSize;
    calcBeamPropsKernel<<<gridSize, blockSize>>>(d_b, beamCount);
    cudaDeviceSynchronize();
    auto e_kernel = SteadyClock::now();

    // 4. D2H
    auto s_d2h = SteadyClock::now();
    cudaMemcpy(beams, d_b, beamCount * sizeof(ShipBeamGpuData), cudaMemcpyDeviceToHost);
    auto e_d2h = SteadyClock::now();

    if (tempAlloc) cudaFree(d_b);
    auto e_total = SteadyClock::now();

    printf("[梁单元 GPU 明细计时 - 数量: %d]\n", beamCount);
    printf("  > 显存获取: %.4f ms\n", get_ms(s_alloc, e_alloc));
    printf("  > 数据上传: %.4f ms\n", get_ms(s_h2d, e_h2d));
    printf("  > 核函数计算: %.4f ms\n", get_ms(s_kernel, e_kernel));
    printf("  > 数据回传: %.4f ms\n", get_ms(s_d2h, e_d2h));
    printf("  > 梁单元总计: %.4f ms\n", get_ms(s_total, e_total));
}

// =========================================================
// 5. GPU 法向量计算接口
// =========================================================

extern "C" void computeGpuNormals(
    const float* vertices,   // host pointer [n_verts * 3]
    const int* indices,      // host pointer [n_tris * 3]
    float* normals,          // host pointer [n_verts * 3] output
    int n_verts,
    int n_tris
) {
    if (!vertices || !indices || !normals || n_verts <= 0 || n_tris <= 0) return;

    float* d_verts = nullptr;
    int* d_idx = nullptr;
    float* d_norms = nullptr;

    size_t v_bytes = n_verts * 3 * sizeof(float);
    size_t i_bytes = n_tris * 3 * sizeof(int);

    cudaMalloc(&d_verts, v_bytes);
    cudaMalloc(&d_idx, i_bytes);
    cudaMalloc(&d_norms, v_bytes);
    cudaMemset(d_norms, 0, v_bytes);

    cudaMemcpy(d_verts, vertices, v_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, indices, i_bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n_tris + blockSize - 1) / blockSize;
    calcTriangleNormalsKernel<<<gridSize, blockSize>>>(d_verts, d_idx, d_norms, n_tris, n_verts);
    cudaDeviceSynchronize();

    gridSize = (n_verts + blockSize - 1) / blockSize;
    normalizeNormalsKernel<<<gridSize, blockSize>>>(d_norms, n_verts);
    cudaDeviceSynchronize();

    cudaMemcpy(normals, d_norms, v_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_verts);
    cudaFree(d_idx);
    cudaFree(d_norms);
}

extern "C" void cleanupGpuMemoryPool() {
    g_memoryPool.cleanup();
}
