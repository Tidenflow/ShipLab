// =========================================================
// ShipGravityLab - 数据模型定义头文件
// 功能：定义船舶结构计算所需的所有数据结构和枚举
// 设计原则：
// 1. 业务实体与GPU数据分离：
//    - ShipPlate/ShipBeam：业务层实体，可包含字符串等非POD字段
//    - ShipPlateGpuData/ShipBeamGpuData：GPU友好结构，仅包含数值字段（POD）
// 2. 内存对齐优化：结构体设计考虑GPU内存访问效率
// 3. 数据类型统一：使用float满足船舶计算精度要求（1e-6）
// =========================================================

#pragma once
#include <string>
#include <vector>

// =========================================================
// 算子系统定义（为未来扩展预留）
// =========================================================

/**
 * @brief 算子类型枚举
 * @note 用于区分适合GPU并行的小算子和需要CPU串行的大算子
 */
enum class OpType {
    SMALL_INDEPENDENT,  // 独立小算子：计算独立，无依赖，适合GPU并行（如单元属性计算）
    LARGE_RELATED       // 强关联大算子：计算间有依赖关系，需要CPU串行执行（如整体装配）
};

/**
 * @brief 算子元数据结构
 * @note 描述算子的属性和依赖关系，用于智能调度
 */
struct OpMeta {
    std::string opName;          // 算子名称（如"calcPlateArea"）
    OpType opType;               // 算子类型（GPU并行或CPU串行）
    std::vector<std::string> deps; // 依赖的拓扑ID（例如先算面积再算重量）
    bool modifyGlobalState;      // 是否修改全局模型状态
    float precision;             // 精度要求（船舶计算通常需要1e-6）
};

// =========================================================
// 板（Plate）数据结构
// 用于表示船舶结构中的板单元（三角形板S3或四边形板S4R）
// =========================================================

/**
 * @brief 板实体结构（业务层）
 * @note 包含完整的业务信息，包括字符串ID，用于业务逻辑处理
 */
struct ShipPlate {
    std::string id;       // 板ID，格式如"plate_123"（来自Abaqus单元ID）
    float area;           // 板面积（平方米），由GPU计算输出
    float thickness;      // 板厚度（米），输入参数，通常为0.01（10mm）
    float volume;         // 板体积（立方米），由GPU计算：体积 = 面积 × 厚度
    float weight;         // 板重量（千克），由GPU计算：重量 = 体积 × 密度
    float centroid[3];    // 板形心坐标（米），三维数组[x, y, z]，由GPU计算
    float density = 7850.0f; // 钢材密度（千克/立方米），默认值7850kg/m³
};

/**
 * @brief 板GPU数据结构（GPU友好，纯数值）
 * @note 专为GPU计算设计，仅包含数值字段（POD），无字符串或动态容器
 *       用于主机-设备数据传输和GPU核函数计算
 */
struct ShipPlateGpuData {
    // ========== 物理属性（输入/输出） ==========
    float area = 0.0f;              // 面积（平方米），GPU计算输出
    float thickness = 0.0f;         // 厚度（米），CPU输入
    float volume = 0.0f;            // 体积（立方米），GPU计算输出
    float weight = 0.0f;            // 重量（千克），GPU计算输出
    float centroid[3] = {0.0f, 0.0f, 0.0f}; // 形心坐标（米），GPU计算输出
    float density = 7850.0f;        // 密度（千克/立方米），CPU输入，默认钢材密度
    
    // ========== 几何信息（输入） ==========
    int nodeCount = 0; // 节点数量：3表示三角形板（S3），4表示四边形板（S4R）
    
    // 顶点坐标数组（每个顶点3个浮点数：x, y, z）
    // 注意：p4仅在四边形板（nodeCount=4）时有效
    float p1[3] = {0.0f, 0.0f, 0.0f}; // 第一个顶点坐标
    float p2[3] = {0.0f, 0.0f, 0.0f}; // 第二个顶点坐标
    float p3[3] = {0.0f, 0.0f, 0.0f}; // 第三个顶点坐标
    float p4[3] = {0.0f, 0.0f, 0.0f}; // 第四个顶点坐标（仅四边形板使用）
};

// =========================================================
// 梁（Beam）数据结构
// 用于表示船舶结构中的梁单元（B31梁单元）
// =========================================================

/**
 * @brief 梁实体结构（业务层）
 * @note 表示船舶结构中的梁（骨）单元，通常是直线段
 */
struct ShipBeam {
    std::string id;       // 梁ID，格式如"beam_456"（来自Abaqus单元ID）
    float length;         // 梁长度（米），由GPU计算输出
    float sectionArea;    // 梁截面积（平方米），输入参数，通常为0.001（10cm²）
    float volume;         // 梁体积（立方米），由GPU计算：体积 = 长度 × 截面积
    float weight;         // 梁重量（千克），由GPU计算：重量 = 体积 × 密度
    float centroid[3];    // 梁形心坐标（米），三维数组[x, y, z]，由GPU计算
    float density = 7850.0f; // 钢材密度（千克/立方米），默认值7850kg/m³
};

/**
 * @brief 梁GPU数据结构（GPU友好，纯数值）
 * @note 专为GPU计算设计，仅包含数值字段
 */
struct ShipBeamGpuData {
    // ========== 物理属性（输入/输出） ==========
    float length = 0.0f;              // 长度（米），GPU计算输出
    float sectionArea = 0.0f;         // 截面积（平方米），CPU输入
    float volume = 0.0f;              // 体积（立方米），GPU计算输出
    float weight = 0.0f;              // 重量（千克），GPU计算输出
    float centroid[3] = {0.0f, 0.0f, 0.0f}; // 形心坐标（米），GPU计算输出
    float density = 7850.0f;          // 密度（千克/立方米），CPU输入
    
    // ========== 几何信息（输入） ==========
    // 梁两端点坐标（每个端点3个浮点数：x, y, z）
    float p1[3] = {0.0f, 0.0f, 0.0f}; // 第一个端点坐标
    float p2[3] = {0.0f, 0.0f, 0.0f}; // 第二个端点坐标
};

// =========================================================
// 渲染数据结构（用于前后端传输）
// =========================================================

/**
 * @brief 三角形渲染单元
 * @note 所有板单元在后端统一三角化后的输出格式
 */
struct RenderTriangle {
    int nodeIds[3];       // 三个顶点的 0-based 索引
    float normal[3];      // 面法向量 (nx, ny, nz)
};
