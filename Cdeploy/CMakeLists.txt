# 指定 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(onnx_deploy C)

# 设置 C 标准
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 设置编译类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 设置编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

# 设置 ONNXRUNTIME 路径（需要根据实际安装路径修改）
set(ONNXRUNTIME_ROOT_PATH "/Users/hyt/Desktop/gtcrn/onnxruntime-osx-x86_64-1.20.1")

# 添加 ONNXRUNTIME 头文件路径
include_directories(
    ${ONNXRUNTIME_ROOT_PATH}/include
)

# 添加 ONNXRUNTIME 库文件路径
link_directories(
    ${ONNXRUNTIME_ROOT_PATH}/lib
)

# 设置输出目录
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 收集源文件
file(GLOB_RECURSE SOURCES 
    "${PROJECT_SOURCE_DIR}/main.c"
)

# 添加可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接 ONNXRUNTIME 库
target_link_libraries(${PROJECT_NAME}
    onnxruntime
)
