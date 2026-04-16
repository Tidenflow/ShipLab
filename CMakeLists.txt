cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------
# 1. 强制指定 CUDA 路径 (防止 CMake 去找不存在的 13.2)
# ---------------------------------------------------------
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS "/usr/local/cuda-12.1/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.1/bin/nvcc")
    endif()
endif()

# 设置 GPU 架构：RTX 4060 对应的架构是 89
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

# ---------------------------------------------------------
# 2. 项目基础配置
# ---------------------------------------------------------
project(ShipGravityLab LANGUAGES CXX CUDA)

# ---------------------------------------------------------
# 3. 查找依赖
# ---------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# --- gRPC / Protobuf（通过系统 apt 安装的版本，使用 pkg-config 路径） ---
find_program(PROTOC_EXE protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
find_library(GRPC_LIB      grpc++   REQUIRED)
find_library(PROTOBUF_LIB  protobuf REQUIRED)

# ---------------------------------------------------------
# 4. 自动运行 protoc 生成 C++ 桩代码
# ---------------------------------------------------------
set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")
file(MAKE_DIRECTORY ${PROTO_OUT_DIR})

set(PROTO_FILE "${PROTO_SRC_DIR}/simulation.proto")
set(PROTO_CC   "${PROTO_OUT_DIR}/simulation.pb.cc")
set(PROTO_H    "${PROTO_OUT_DIR}/simulation.pb.h")
set(GRPC_CC    "${PROTO_OUT_DIR}/simulation.grpc.pb.cc")
set(GRPC_H     "${PROTO_OUT_DIR}/simulation.grpc.pb.h")

add_custom_command(
    OUTPUT ${PROTO_CC} ${PROTO_H} ${GRPC_CC} ${GRPC_H}
    COMMAND ${PROTOC_EXE}
        --proto_path=${PROTO_SRC_DIR}
        --cpp_out=${PROTO_OUT_DIR}
        --grpc_out=${PROTO_OUT_DIR}
        --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
        ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "[Protoc] 生成 gRPC C++ 桩代码..."
)

add_library(proto_gen_lib STATIC ${PROTO_CC} ${GRPC_CC})
target_include_directories(proto_gen_lib PUBLIC
    ${PROTO_OUT_DIR}
    /usr/include          # grpc/grpc++
    /usr/include/grpc++
)
target_link_libraries(proto_gen_lib PUBLIC
    ${GRPC_LIB}
    ${PROTOBUF_LIB}
)

# ---------------------------------------------------------
# 5. 原始可执行文件（本地测试用）
# ---------------------------------------------------------
add_executable(${PROJECT_NAME}
    src/cli/main.cu
    src/core/GpuOperatorOptimized_v2.cu
    src/core/CpuAggregator.cpp
    src/io/MemoryMappedFileReader.h
    src/core/OperatorTags.h
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart
)

# ---------------------------------------------------------
# 6. gRPC 计算后端服务器
# ---------------------------------------------------------
add_executable(ShipGravityServer
    src/server/main_server.cpp
    src/core/GpuOperatorOptimized_v2.cu
    src/core/CpuAggregator.cpp
    src/io/MemoryMappedFileReader.h
    src/core/OperatorTags.h
)

target_include_directories(ShipGravityServer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/io
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float/include
    ${PROTO_OUT_DIR}
    /usr/include
    /usr/include/grpc++
)

target_link_libraries(ShipGravityServer PRIVATE
    CUDA::cudart
    proto_gen_lib
    ${GRPC_LIB}
    ${PROTOBUF_LIB}
    pthread
)

# ---------------------------------------------------------
# 7. 编译选项优化（同时适用于两个目标）
# ---------------------------------------------------------
foreach(_target ${PROJECT_NAME} ShipGravityServer)
    if(MSVC)
        target_compile_options(${_target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2 /utf-8>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8 -O2 -use_fast_math>
        )
    else()
        target_compile_options(${_target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math -Xptxas=-v --expt-relaxed-constexpr>
        )
    endif()
    target_compile_definitions(${_target} PRIVATE
        SHIPGRAVITYLAB_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endforeach()