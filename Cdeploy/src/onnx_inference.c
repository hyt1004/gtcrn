#include "include/onnx_inference.h"
#include <stdio.h>
#include <stdlib.h>

int init_onnx_inference(OnnxInference* inference, const char* model_path) {
    inference->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    
    // 创建环境
    if (inference->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_inference", &inference->env) != NULL) {
        printf("创建ONNX环境失败\n");
        return -1;
    }

    // 创建会话选项
    if (inference->api->CreateSessionOptions(&inference->session_options) != NULL) {
        printf("创建会话选项失败\n");
        return -1;
    }

    // 创建会话
    if (inference->api->CreateSession(inference->env, model_path, inference->session_options, 
                                    &inference->session) != NULL) {
        printf("创建ONNX会话失败\n");
        return -1;
    }

    // 获取默认分配器
    if (inference->api->GetAllocatorWithDefaultOptions(&inference->allocator) != NULL) {
        printf("获取内存分配器失败\n");
        return -1;
    }

    return 0;
}

int run_inference(OnnxInference* inference, 
                 float* input_data, 
                 const int64_t* input_dims, 
                 size_t input_dim_count,
                 float* output_data,
                 size_t output_size) {
    OrtMemoryInfo* memory_info = NULL;
    OrtValue* input_tensor = NULL;
    OrtValue* output_tensor = NULL;
    
    // 创建内存信息
    inference->api->CreateMemoryInfo("Cpu", OrtArenaAllocator, OrtDeviceAllocator, 0, &memory_info);

    // 创建输入tensor
    inference->api->CreateTensorWithDataAsOrtValue(memory_info, input_data, 
        input_dim_count * sizeof(float), input_dims, input_dim_count, 
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);

    // 获取输入和输出名称
    char* input_name;
    char* output_name;
    inference->api->SessionGetInputName(inference->session, 0, inference->allocator, &input_name);
    inference->api->SessionGetOutputName(inference->session, 0, inference->allocator, &output_name);

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    // 运行推理
    inference->api->Run(inference->session, NULL, input_names, (const OrtValue* const*)&input_tensor, 
                       1, output_names, 1, &output_tensor);

    // 获取输出数据
    float* output_tensor_data;
    inference->api->GetTensorMutableData(output_tensor, (void**)&output_tensor_data);
    
    // 复制输出数据
    memcpy(output_data, output_tensor_data, output_size * sizeof(float));

    // 释放资源
    inference->api->ReleaseValue(input_tensor);
    inference->api->ReleaseValue(output_tensor);
    inference->api->ReleaseMemoryInfo(memory_info);
    inference->api->AllocatorFree(inference->allocator, input_name);
    inference->api->AllocatorFree(inference->allocator, output_name);

    return 0;
}

void free_onnx_inference(OnnxInference* inference) {
    if (inference->session != NULL) {
        inference->api->ReleaseSession(inference->session);
    }
    if (inference->session_options != NULL) {
        inference->api->ReleaseSessionOptions(inference->session_options);
    }
    if (inference->env != NULL) {
        inference->api->ReleaseEnv(inference->env);
    }
}