#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "onnxruntime_c_api.h"

// 错误检查宏 - 用于检查一般错误状态
// 如果status不为NULL,打印错误信息并退出程序
#define CHECK(status) \
    if (status != NULL) { \
        printf("Error: %s\n", status); \
        exit(1); \
    }

// 检查 ORT API 状态 - 用于检查ONNX Runtime API调用的返回状态
// 如果status不是ORT_OK,打印错误信息并退出程序
#define CHECK_STATUS(status) \
    if (status != ORT_OK) { \
        printf("Error: ORT API 调用失败\n"); \
        exit(1); \
    }

/**
 * 打印张量(Tensor)的详细信息
 * @param ort_api ONNX Runtime API接口指针
 * @param session 模型会话指针
 * @param index 张量索引
 * @param type 张量类型("input"或"output")
 */
void print_tensor_info(const OrtApi* ort_api, OrtSession* session, size_t index, const char* type) {
    OrtTypeInfo* type_info;                    // 类型信息
    const OrtTensorTypeAndShapeInfo* tensor_info;  // 张量形状和类型信息
    size_t dim_count;                          // 维度数量
    int64_t* dims;                            // 各维度大小
    ONNXTensorElementDataType tensor_type;     // 张量数据类型
    char* name;                               // 张量名称
    OrtAllocator* allocator;                  // 内存分配器
    
    // 获取默认内存分配器
    CHECK_STATUS(ort_api->GetAllocatorWithDefaultOptions(&allocator));
    
    // 根据类型获取节点名称
    if (strcmp(type, "input") == 0) {
        CHECK_STATUS(ort_api->SessionGetInputName(session, index, allocator, &name));
    } else {
        CHECK_STATUS(ort_api->SessionGetOutputName(session, index, allocator, &name));
    }
    
    // 获取节点的类型信息
    if (strcmp(type, "input") == 0) {
        CHECK_STATUS(ort_api->SessionGetInputTypeInfo(session, index, &type_info));
    } else {
        CHECK_STATUS(ort_api->SessionGetOutputTypeInfo(session, index, &type_info));
    }
    
    // 将类型信息转换为张量信息
    CHECK_STATUS(ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info));
    
    // 获取张量的维度信息
    CHECK_STATUS(ort_api->GetDimensionsCount(tensor_info, &dim_count));
    dims = malloc(sizeof(int64_t) * dim_count);
    CHECK_STATUS(ort_api->GetDimensions(tensor_info, dims, dim_count));
    
    // 获取张量的数据类型
    CHECK_STATUS(ort_api->GetTensorElementType(tensor_info, &tensor_type));
    
    // 打印张量的详细信息
    printf("\n%s %zu:\n", type, index);
    printf("  名称: %s\n", name);
    printf("  形状: [");
    for(size_t i = 0; i < dim_count; i++) {
        printf("%ld", dims[i]);
        if (i < dim_count - 1) printf(", ");
    }
    printf("]\n");
    printf("  数据类型: ");
    switch(tensor_type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   printf("float32\n"); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  printf("float64\n"); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   printf("int32\n"); break;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   printf("int64\n"); break;
        default: printf("其他类型(%d)\n", tensor_type);
    }
    
    // 释放分配的资源
    free(dims);
    ort_api->ReleaseTypeInfo(type_info);
    ort_api->AllocatorFree(allocator, name);
}

/**
 * 主函数 - 演示如何使用ONNX Runtime C API加载和运行ONNX模型
 */
int main(void) {
    // 获取ONNX Runtime API接口
    const OrtApi* ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* env = NULL;           // ONNX Runtime环境
    OrtSession* session = NULL;    // 模型会话
    OrtSessionOptions* session_options = NULL;  // 会话选项
    
    // 创建ONNX Runtime环境
    CHECK_STATUS(ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));
    
    // 创建会话选项
    CHECK_STATUS(ort_api->CreateSessionOptions(&session_options));
    printf("CreateSessionOptions\n");
    
    // 设置线程数为1
    CHECK_STATUS(ort_api->SetIntraOpNumThreads(session_options, 1));
    printf("SetIntraOpNumThreads: 1\n");
    
    // 加载ONNX模型文件
    const char* model_path = "/Users/hyt/Desktop/gtcrn/stream/onnx_models/gtcrn_simple.onnx";
    CHECK_STATUS(ort_api->CreateSession(env, model_path, session_options, &session));
    printf("CreateSession\n");

    // 获取模型的输入输出节点信息
    size_t num_input_nodes, num_output_nodes;
    CHECK_STATUS(ort_api->SessionGetInputCount(session, &num_input_nodes));
    CHECK_STATUS(ort_api->SessionGetOutputCount(session, &num_output_nodes));
    
    // 打印输入节点信息
    printf("\n输入节点数量: %zu\n", num_input_nodes);
    for(size_t i = 0; i < num_input_nodes; i++) {
        print_tensor_info(ort_api, session, i, "input");
    }
    
    // 打印输出节点信息
    printf("\n输出节点数量: %zu\n", num_output_nodes);
    for(size_t i = 0; i < num_output_nodes; i++) {
        print_tensor_info(ort_api, session, i, "output");
    }

    /* 模型的输入输出信息:
    输入节点数量: 4
    input 0: 名称: mix 形状: [1, 257, 1, 2] 数据类型: float32
    input 1: 名称: conv_cache 形状: [2, 1, 16, 16, 33] 数据类型: float32
    input 2: 名称: tra_cache 形状: [2, 3, 1, 1, 16] 数据类型: float32
    input 3: 名称: inter_cache 形状: [2, 1, 33, 16] 数据类型: float32

    输出节点数量: 4
    output 0: 名称: enh 形状: [1, 257, 1, 2] 数据类型: float32
    output 1: 名称: conv_cache_out 形状: [2, 1, 16, 16, 33] 数据类型: float32
    output 2: 名称: tra_cache_out 形状: [2, 3, 1, 1, 16] 数据类型: float32
    output 3: 名称: inter_cache_out 形状: [2, 1, 33, 16] 数据类型: float32
    */

    
    // 准备内存分配器和内存信息对象
    OrtAllocator* allocator=NULL;
    OrtMemoryInfo* memory_info=NULL;
    CHECK_STATUS(ort_api->GetAllocatorWithDefaultOptions(&allocator));
    CHECK_STATUS(ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeCPU, &memory_info));

    // 为输入张量分配内存空间
    float mix_data[257 * 1 * 2];                    // 混合信号数据 [1, 257, 1, 2]
    float conv_cache_data[2 * 1 * 16 * 16 * 33];    // 卷积缓存数据 [2, 1, 16, 16, 33]
    float tra_cache_data[2 * 3 * 1 * 1 * 16];       // 时序注意力缓存数据 [2, 3, 1, 1, 16]
    float inter_cache_data[2 * 1 * 33 * 16];        // 中间缓存数据 [2, 1, 33, 16]

    // 初始化输入数据(示例数据)
    for(int i = 0; i < 257 * 1 * 2; i++) mix_data[i] = 0.1f;
    for(int i = 0; i < 2 * 1 * 16 * 16 * 33; i++) conv_cache_data[i] = 0.0f;
    for(int i = 0; i < 2 * 3 * 1 * 1 * 16; i++) tra_cache_data[i] = 0.0f;
    for(int i = 0; i < 2 * 1 * 33 * 16; i++) inter_cache_data[i] = 0.0f;

    // 定义各输入张量的形状
    const int64_t mix_shape[] = {1, 257, 1, 2};
    const int64_t conv_cache_shape[] = {2, 1, 16, 16, 33};
    const int64_t tra_cache_shape[] = {2, 3, 1, 1, 16};
    const int64_t inter_cache_shape[] = {2, 1, 33, 16};

    // 创建输入张量对象
    OrtValue* input_tensors[4] = {NULL, NULL, NULL, NULL};
    CHECK_STATUS(ort_api->CreateTensorWithDataAsOrtValue(
        memory_info, mix_data, sizeof(mix_data),
        mix_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[0]));
    CHECK_STATUS(ort_api->CreateTensorWithDataAsOrtValue(
        memory_info, conv_cache_data, sizeof(conv_cache_data),
        conv_cache_shape, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[1]));
    CHECK_STATUS(ort_api->CreateTensorWithDataAsOrtValue(
        memory_info, tra_cache_data, sizeof(tra_cache_data),
        tra_cache_shape, 5, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[2]));
    CHECK_STATUS(ort_api->CreateTensorWithDataAsOrtValue(
        memory_info, inter_cache_data, sizeof(inter_cache_data),
        inter_cache_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors[3]));

    // 获取输入输出节点的名称
    const char* input_names[4];
    const char* output_names[4];
    char *name_temp;
    for(size_t i = 0; i < 4; i++) {
        CHECK_STATUS(ort_api->SessionGetInputName(session, i, allocator, &name_temp));
        input_names[i] = name_temp;
        CHECK_STATUS(ort_api->SessionGetOutputName(session, i, allocator, &name_temp));
        output_names[i] = name_temp;
    }

    // 准备输出张量数组
    OrtValue* output_tensors[4] = {NULL, NULL, NULL, NULL};

    // 运行模型推理
    CHECK_STATUS(ort_api->Run(session, NULL, input_names, 
        (const OrtValue* const*)input_tensors, 4, output_names, 4, output_tensors));

    // 获取输出数据指针
    // 定义输出数据指针
    float *enh_output, *conv_cache_output, *tra_cache_output, *inter_cache_output;
    CHECK_STATUS(ort_api->GetTensorMutableData(output_tensors[0], (void**)&enh_output));
    CHECK_STATUS(ort_api->GetTensorMutableData(output_tensors[1], (void**)&conv_cache_output));
    CHECK_STATUS(ort_api->GetTensorMutableData(output_tensors[2], (void**)&tra_cache_output));
    CHECK_STATUS(ort_api->GetTensorMutableData(output_tensors[3], (void**)&inter_cache_output));

    // 打印部分增强后的数据作为示例
    printf("\n增强后的前几个数据:\n");
    printf("\nenh_output:  ");
    for(int i = 0; i < 5; i++) {
        printf("%f ", enh_output[i]);
    }
    printf("\nconv_cache_output:  ");
    for(int i = 0; i < 5; i++) {
        printf("%f ", conv_cache_output[i]);
    }
    printf("\ntra_cache_output:  ");
    for(int i = 0; i < 5; i++) {
        printf("%f ", tra_cache_output[i]);
    }
    printf("\ninter_cache_output:  ");
    for(int i = 0; i < 5; i++) {
        printf("%f ", inter_cache_output[i]);
    }
    printf("\n");

    // 清理所有分配的资源
    for(int i = 0; i < 4; i++) {
        if(input_tensors[i]) ort_api->ReleaseValue(input_tensors[i]);
        if(output_tensors[i]) ort_api->ReleaseValue(output_tensors[i]);
        ort_api->AllocatorFree(allocator, (void*)input_names[i]);
        ort_api->AllocatorFree(allocator, (void*)output_names[i]);
    }
    ort_api->ReleaseMemoryInfo(memory_info);
    ort_api->ReleaseSession(session);
    ort_api->ReleaseSessionOptions(session_options);
    ort_api->ReleaseEnv(env);
    
    return 0;
}