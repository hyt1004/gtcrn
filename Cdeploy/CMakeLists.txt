cmake_minimum_required(VERSION 3.10)
project(onnx_inference_demo)

# 设置ONNX Runtime的路径
set(ONNXRUNTIME_ROOT_PATH "/Users/hyt/Desktop/gtcrn/onnxruntime-osx-x86_64-1.20.1")

# 设置ONNX Runtime的头文件路径
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT_PATH}/include")

# 设置ONNX Runtime的库文件路径
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.dylib")

# 创建库
add_library(onnx_inference
    include/onnx_inference.h
    src/onnx_inference.c
)

# 设置包含目录
target_include_directories(onnx_inference
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# 链接ONNX Runtime
target_link_libraries(onnx_inference
    PUBLIC
    ${ONNXRUNTIME_LIB}
)

# 创建可执行文件
add_executable(inference_demo main.c)
target_link_libraries(inference_demo
    PRIVATE
    onnx_inference
)

# 确保运行时能找到动态库
set_target_properties(inference_demo PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ONNXRUNTIME_ROOT_PATH}/lib"
)