#include "include/onnx_inference.h"
#include <stdio.h>

int main() {
    OnnxInference inference = {0};  // 初始化所有成员为0
    
    // 初始化推理环境
    if (init_onnx_inference(&inference, "path/to/your/model.onnx") != 0) {
        printf("初始化ONNX推理环境失败\n");
        return -1;
    }

    // 准备输入数据（示例）
    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int64_t input_dims[] = {1, 4};  // 批次大小为1，输入特征维度为4
    float output_data[2];  // 假设输出是2个数字

    // 运行推理
    if (run_inference(&inference, input_data, input_dims, 2, output_data, 2) != 0) {
        printf("模型推理失败\n");
        free_onnx_inference(&inference);
        return -1;
    }

    // 打印结果
    printf("推理结果: %f, %f\n", output_data[0], output_data[1]);

    // 释放资源
    free_onnx_inference(&inference);
    return 0;
}