// =========================================================
// ShipGravityLab - CPU汇总模块
// 功能：将所有单元（板和梁）的重量和形心汇总为整体重心
// 职责：
// 1. 累加所有单元的重量
// 2. 计算总重量矩（重量 × 形心坐标）
// 3. 计算整体重心坐标（总重量矩 / 总重量）
// 性能：当前耗时约0.65ms，不是性能瓶颈
// =========================================================

#include "OperatorTags.h"

/**
 * @brief 计算船舶结构的总重量和总重心
 * @param plates 板数据数组（GPU计算结果）
 * @param plateCount 板数量
 * @param beams 梁数据数组（GPU计算结果）
 * @param beamCount 梁数量
 * @param totalWeight 输出参数：总重量（千克）
 * @param totalCentroid 输出参数：总重心坐标（米），三维数组[x, y, z]
 * @note 数学原理：
 *       1. 总重量 = Σ(每个单元重量)
 *       2. 总重量矩 = Σ(每个单元重量 × 该单元形心坐标)
 *       3. 总重心坐标 = 总重量矩 / 总重量
 *       如果总重量为0，重心坐标保持为[0,0,0]
 */
void calcTotalGravityCenter(
    const ShipPlateGpuData* plates, int plateCount,
    const ShipBeamGpuData* beams, int beamCount,
    float& totalWeight, float totalCentroid[3]) {
    
    // ========== 初始化累加器 ==========
    totalWeight = 0;                    // 总重量
    totalCentroid[0] = 0;               // X方向总重量矩
    totalCentroid[1] = 0;               // Y方向总重量矩
    totalCentroid[2] = 0;               // Z方向总重量矩
    
    // ========== 累加所有板的贡献 ==========
    for (int i = 0; i < plateCount; i++) {
        totalWeight += plates[i].weight; // 累加重量
        // 累加重量矩：重量 × 形心坐标
        totalCentroid[0] += plates[i].weight * plates[i].centroid[0]; // X方向
        totalCentroid[1] += plates[i].weight * plates[i].centroid[1]; // Y方向
        totalCentroid[2] += plates[i].weight * plates[i].centroid[2]; // Z方向
    }
    
    // ========== 累加所有梁的贡献 ==========
    for (int i = 0; i < beamCount; i++) {
        totalWeight += beams[i].weight; // 累加重量
        // 累加重量矩：重量 × 形心坐标
        totalCentroid[0] += beams[i].weight * beams[i].centroid[0]; // X方向
        totalCentroid[1] += beams[i].weight * beams[i].centroid[1]; // Y方向
        totalCentroid[2] += beams[i].weight * beams[i].centroid[2]; // Z方向
    }
    
    // ========== 计算整体重心坐标 ==========
    // 整体重心 = 总重量矩 / 总重量
    if (totalWeight != 0.0f) {
        totalCentroid[0] /= totalWeight; // X坐标
        totalCentroid[1] /= totalWeight; // Y坐标
        totalCentroid[2] /= totalWeight; // Z坐标
    }
    // 注意：如果总重量为0，重心坐标保持为[0,0,0]
}