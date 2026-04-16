// =========================================================
// MemoryMappedFileReader - 内存映射文件读取器 (跨平台版)
// 功能：高速读取 Abaqus/INP 网格文件，解析节点 + 板/梁单元
// 优化：无 push_back、预分配内存、两遍扫描、零拷贝、极致性能
// 适用：大型有限元网格解析（百万级单元无压力）
// =========================================================

#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cmath>

// --- 跨平台兼容性处理：识别 Windows 或 Linux ---
#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <sys/mman.h>  // Linux 内存映射核心头文件
    #include <sys/stat.h>  // 获取文件大小
    #include <fcntl.h>     // open 函数
    #include <unistd.h>    // close 函数
#endif

#include "OperatorTags.h"

// =========================================================
// 业务实体结构（与 INP 文件格式一一对应）
// =========================================================

/**
 * @brief INP 文件节点结构体
 * 存储：节点ID + 三维坐标 (x,y,z)
 */
struct InpNode {
    int id = 0;             // 节点编号
    float x = 0.0f;        // X 坐标
    float y = 0.0f;        // Y 坐标
    float z = 0.0f;        // Z 坐标

    InpNode(int id_, float x_, float y_, float z_) : id(id_), x(x_), y(y_), z(z_) {}
    InpNode() = default;
};

/**
 * @brief 板单元业务实体（原始解析数据）
 */
struct PlateEntity {
    int id = 0;                         // 单元编号
    int nodeIds[4] = {0, 0, 0, 0};      // 节点编号（最多4个）
    int nodeCount = 0;                  // 节点数量（3=三角形，4=四边形）
    float thickness = 0.01f;           // 板厚度（默认1cm）
    float density = 7850.0f;            // 材料密度（默认钢）
};

/**
 * @brief 梁单元业务实体（原始解析数据）
 */
struct BeamEntity {
    int id = 0;                         // 单元编号
    int nodeIds[2] = {0, 0};            // 梁的两个节点
    float sectionArea = 0.0001f;        // 截面面积
    float density = 7850.0f;            // 材料密度
};

// =========================================================
// 内存映射读取器核心类
// =========================================================
class MemoryMappedFileReader {
private:
    // 跨平台文件句柄/描述符
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMapping = NULL;
#else
    int fd = -1;  // Linux 下的文件描述符
#endif

    // 文件内存指针
    const char* fileData = nullptr;        // 文件映射到内存的起始地址
    size_t fileSize = 0;                   // 文件总字节数
    const char* current = nullptr;         // 当前解析位置指针
    const char* endPtr = nullptr;          // 文件结束指针

    // 外部数据容器（由主函数传入，解析结果直接写入）
    std::vector<InpNode>* nodes = nullptr;
    std::vector<PlateEntity>* plates = nullptr;
    std::vector<BeamEntity>* beams = nullptr;

    // 解析状态标记
    bool inNodeSection = false;           // 是否在 *NODE 节点段
    bool inElementSection = false;         // 是否在 *ELEMENT 单元段
    std::string currentElementType;        // 当前单元类型（S3/S4R/B31）

    // 单元数量统计（第一遍扫描得出，用于预分配）
    size_t totalNodes = 0;
    size_t totalPlates = 0;
    size_t totalBeams = 0;

    // 节点ID → 数组下标 映射表（加速坐标查找）
    std::vector<int> nodeIdToIndex;

    // 跨平台强制内联宏
#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline __attribute__((always_inline))
#endif

    // 超快跳过分隔符（兼容 INP 格式：值后可能有尾随空格，再跟逗号，再跟前导空格）
    static FORCE_INLINE void skipSeparators(const char*& p) {
        while (*p == ' ' || *p == '\t') p++;  // 跳逗号前的空格（如 "7.4   ,"）
        if (*p == ',') p++;                    // 跳逗号
        while (*p == ' ' || *p == '\t') p++;  // 跳逗号后的空格
    }

public:
    // ====================== 核心输出 ======================
    // 直接生成好的 GPU 数据结构，主函数可直接送入显卡计算
    std::vector<ShipPlateGpuData> gpuPlates;
    std::vector<ShipBeamGpuData> gpuBeams;

public:
    MemoryMappedFileReader() = default;
    ~MemoryMappedFileReader() { close(); }

    /**
     * @brief 打开文件并创建内存映射 (支持 Win/Linux)
     */
    bool open(const std::string& filename) {
#ifdef _WIN32
        hFile = CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER size;
        if (!GetFileSizeEx(hFile, &size)) { CloseHandle(hFile); return false; }
        fileSize = static_cast<size_t>(size.QuadPart);

        hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMapping) { CloseHandle(hFile); return false; }

        fileData = static_cast<const char*>(MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0));
#else
        fd = ::open(filename.c_str(), O_RDONLY);
        if (fd == -1) return false;

        struct stat st;
        if (fstat(fd, &st) == -1) { ::close(fd); return false; }
        fileSize = st.st_size;

        fileData = static_cast<const char*>(mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0));
        if (fileData == MAP_FAILED) { ::close(fd); fileData = nullptr; return false; }
#endif
        if (!fileData) return false;
        current = fileData;
        endPtr = fileData + fileSize;
        return true;
    }

    /**
     * @brief 关闭映射并释放资源（RAII 自动调用）
     */
    void close() {
#ifdef _WIN32
        if (fileData) UnmapViewOfFile(fileData);
        if (hMapping) CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
        hMapping = NULL; hFile = INVALID_HANDLE_VALUE;
#else
        if (fileData) munmap(const_cast<char*>(fileData), fileSize);
        if (fd != -1) ::close(fd);
        fd = -1;
#endif
        fileData = nullptr;
    }

    /**
     * @brief 对外解析接口
     * 采用【两遍扫描法】：
     * 第1遍：只统计数量 → 第2遍：直接写入预分配内存 → 无push_back、无扩容
     */
    bool parse(std::vector<InpNode>& outNodes,
               std::vector<PlateEntity>& outPlates,
               std::vector<BeamEntity>& outBeams) {
        if (!fileData) return false;
        nodes = &outNodes; plates = &outPlates; beams = &outBeams;

        // ========== 第1遍：极快扫描，只统计节点/单元数量 ==========
        countElements();

        // ========== 一次性预分配所有内存 ==========
        nodes->resize(totalNodes);
        plates->resize(totalPlates);
        gpuPlates.resize(totalPlates);
        beams->resize(totalBeams);
        gpuBeams.resize(totalBeams);

        // 初始化节点ID映射表（根据最大节点ID动态增长）
        nodeIdToIndex.clear();
        nodeIdToIndex.resize(1000000, -1); 

        // ========== 第2遍：正式解析，直接写入内存，零拷贝 ==========
        current = fileData;
        inNodeSection = false; inElementSection = false;
        size_t nodeIdx = 0, plateIdx = 0, beamIdx = 0;

        while (current < endPtr) {
            while (current < endPtr && (*current <= ' ')) ++current;
            if (current >= endPtr) break;

            const char* lineStart = current;
            while (current < endPtr && *current != '\n' && *current != '\r') ++current;
            size_t lineLen = current - lineStart;

            if (*lineStart == '*') {
                processSectionStart(lineStart, lineLen);
                continue;
            }

            if (inNodeSection && nodeIdx < totalNodes) {
                parseNodeLine(lineStart, lineLen, (*nodes)[nodeIdx], static_cast<int>(nodeIdx));
                ++nodeIdx;
            } else if (inElementSection) {
                if ((currentElementType == "S3" || currentElementType == "S4R") && plateIdx < totalPlates) {
                    parsePlateLine(lineStart, lineLen, (*plates)[plateIdx], gpuPlates[plateIdx]);
                    ++plateIdx;
                } else if (currentElementType == "B31" && beamIdx < totalBeams) {
                    parseBeamLine(lineStart, lineLen, (*beams)[beamIdx], gpuBeams[beamIdx]);
                    ++beamIdx;
                }
            }
        }
        return true;
    }

private:
    /**
     * @brief 第一遍扫描：只统计数量，不解析内容（极快）
     */
    void countElements() {
        totalNodes = totalPlates = totalBeams = 0;
        const char* p = fileData;
        bool nodeSec = false, elemSec = false;
        std::string elemType;

        while (p < endPtr) {
            while (p < endPtr && *p <= ' ') ++p;
            if (p >= endPtr) break;
            const char* line = p;
            while (p < endPtr && *p != '\n' && *p != '\r') ++p;
            size_t len = p - line;

            if (len < 2 || (line[0] == '*' && line[1] == '*')) continue;
            if (*line == '*') {
                if (len >= 5 && !memcmp(line + 1, "NODE", 4)) { nodeSec = true; elemSec = false; }
                else if (len >= 8 && !memcmp(line + 1, "ELEMENT", 7)) {
                    nodeSec = false; elemSec = true;
                    const char* typePos = static_cast<const char*>(cross_memmem(line, len, "TYPE=", 5));
                    if (typePos) {
                        const char* ts = typePos + 5; const char* te = ts;
                        while (te < line + len && *te != ',' && *te != ' ') ++te;
                        elemType = std::string(ts, te);
                        for (char& c : elemType) c = toupper(c);
                    }
                } else { nodeSec = elemSec = false; }
                continue;
            }
            if (nodeSec) ++totalNodes;
            else if (elemSec) {
                if (elemType == "S3" || elemType == "S4R") ++totalPlates;
                else if (elemType == "B31") ++totalBeams;
            }
        }
    }

    void processSectionStart(const char* line, size_t len) {
        if (len >= 5 && !memcmp(line + 1, "NODE", 4)) {
            inNodeSection = true; inElementSection = false;
        } else if (len >= 8 && !memcmp(line + 1, "ELEMENT", 7)) {
            inNodeSection = false; inElementSection = true;
            const char* typePos = static_cast<const char*>(cross_memmem(line, len, "TYPE=", 5));
            if (typePos) {
                const char* ts = typePos + 5; const char* te = ts;
                while (te < line + len && *te != ',' && *te != ' ') ++te;
                currentElementType = std::string(ts, te);
                for (char& c : currentElementType) c = toupper(c);
            }
        } else { inNodeSection = inElementSection = false; }
    }

    /**
     * @brief 超快解析节点行
     */
    void parseNodeLine(const char* line, size_t len, InpNode& out, int idx) {
        const char *p = line, *end = line + len;
        int id = parseInt(p, end); skipSeparators(p);
        float x = parseFloat(p, end); skipSeparators(p);
        float y = parseFloat(p, end); skipSeparators(p);
        float z = parseFloat(p, end);
        out = InpNode(id, x, y, z);
        if (id >= (int)nodeIdToIndex.size()) nodeIdToIndex.resize(id + 10000, -1);
        nodeIdToIndex[id] = idx;
    }

    /**
     * @brief 超快解析板单元行并同步填充 GPU 数据结构
     */
    void parsePlateLine(const char* line, size_t len, PlateEntity& plate, ShipPlateGpuData& gpu) {
        const char *p = line, *end = line + len;
        int elemId = parseInt(p, end); if (elemId <= 0) return;
        int n = (currentElementType == "S3") ? 3 : 4;
        plate.id = elemId; plate.nodeCount = n;
        gpu.nodeCount = n; gpu.thickness = 0.01f; gpu.density = 7850.0f;
        int ids[4] = {0};
        for (int i = 0; i < n; ++i) { skipSeparators(p); ids[i] = parseInt(p, end); plate.nodeIds[i] = ids[i]; }
        
        auto getG = [&](int id) { return (*nodes)[nodeIdToIndex[id]]; };
        InpNode n0 = getG(ids[0]), n1 = getG(ids[1]), n2 = getG(ids[2]);
        gpu.p1[0] = n0.x; gpu.p1[1] = n0.y; gpu.p1[2] = n0.z;
        gpu.p2[0] = n1.x; gpu.p2[1] = n1.y; gpu.p2[2] = n1.z;
        gpu.p3[0] = n2.x; gpu.p3[1] = n2.y; gpu.p3[2] = n2.z;
        if (n >= 4) { InpNode n3 = getG(ids[3]); gpu.p4[0] = n3.x; gpu.p4[1] = n3.y; gpu.p4[2] = n3.z; }
    }

    /**
     * @brief 超快解析梁单元行
     */
    void parseBeamLine(const char* line, size_t len, BeamEntity& beam, ShipBeamGpuData& gpu) {
        const char *p = line, *end = line + len;
        int elemId = parseInt(p, end); if (elemId <= 0) return;
        beam.id = elemId; gpu.sectionArea = 0.0001f; gpu.density = 7850.0f;
        int ids[2]; 
        for (int i = 0; i < 2; ++i) { skipSeparators(p); ids[i] = parseInt(p, end); beam.nodeIds[i] = ids[i]; }
        auto getG = [&](int id) { return (*nodes)[nodeIdToIndex[id]]; };
        InpNode n0 = getG(ids[0]), n1 = getG(ids[1]);
        gpu.p1[0] = n0.x; gpu.p1[1] = n0.y; gpu.p1[2] = n0.z;
        gpu.p2[0] = n1.x; gpu.p2[1] = n1.y; gpu.p2[2] = n1.z;
    }

    /**
     * @brief 超快整数解析（无库函数依赖）
     */
    inline int parseInt(const char*& p, const char* end) {
        int v = 0; bool neg = false;
        while (p < end && *p <= ' ') ++p;
        if (p < end && *p == '-') { neg = true; ++p; }
        while (p < end && *p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); ++p; }
        return neg ? -v : v;
    }

    /**
     * @brief 超快浮点数解析（支持科学计数法）
     */
    inline float parseFloat(const char*& p, const char* end) {
        float v = 0.0f; bool neg = false;
        while (p < end && *p <= ' ') ++p;
        if (p < end && *p == '-') { neg = true; ++p; }
        while (p < end && *p >= '0' && *p <= '9') { v = v * 10.0f + (*p - '0'); ++p; }
        if (p < end && *p == '.') {
            ++p; float f = 0.1f;
            while (p < end && *p >= '0' && *p <= '9') { v += (*p - '0') * f; f *= 0.1f; ++p; }
        }
        if (p < end && (*p == 'e' || *p == 'E')) {
            ++p; bool eneg = false; int exp = 0;
            if (*p == '-') { eneg = true; ++p; } else if (*p == '+') ++p;
            while (p < end && *p >= '0' && *p <= '9') { exp = exp * 10 + (*p - '0'); ++p; }
            v *= (eneg ? (float)std::pow(10, -exp) : (float)std::pow(10, exp));
        }
        return neg ? -v : v;
    }

    /**
     * @brief 跨平台内存查找（用于识别 TYPE=）
     */
    const void* cross_memmem(const void* h, size_t hl, const void* n, size_t nl) {
        if (nl == 0) return h;
        if (hl < nl) return nullptr;
        const char* hh = (const char*)h; const char* nn = (const char*)n;
        for (size_t i = 0; i <= hl - nl; ++i) if (!memcmp(hh + i, nn, nl)) return hh + i;
        return nullptr;
    }
};