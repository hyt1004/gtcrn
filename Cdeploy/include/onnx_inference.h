#ifndef ONNX_INFERENCE_H
#define ONNX_INFERENCE_H

#include <onnxruntime_c_api.h>

// 定义推理上下文结构体
typedef struct {
    OrtEnv* env;                    // ONNX运行环境
    OrtSession* session;            // ONNX会话
    OrtSessionOptions* session_options;  // 会话选项
    OrtAllocator* allocator;        // 内存分配器
    const OrtApi* api;              // ONNX Runtime API
} OnnxInference;

// 初始化ONNX推理环境
int init_onnx_inference(OnnxInference* inference, const char* model_path);

// 执行模型推理
// input_data: 输入数据
// input_dims: 输入维度数组
// input_dim_count: 输入维度数量
// output_data: 输出数据缓冲区
// output_size: 输出数据大小
int run_inference(OnnxInference* inference, 
                 float* input_data, 
                 const int64_t* input_dims, 
                 size_t input_dim_count,
                 float* output_data,
                 size_t output_size);

// 释放ONNX推理环境资源
void free_onnx_inference(OnnxInference* inference);

#endif // ONNX_INFERENCE_H