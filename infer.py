import os
import torch
import soundfile as sf
from gtcrn import GTCRN


## load model
device = torch.device("cpu")
model = GTCRN().eval()
ckpt = torch.load(os.path.join('checkpoints', 'model_trained_on_dns3.tar'), map_location=device)
model.load_state_dict(ckpt['model'])

## load data
# 读取音频文件
mix, fs = sf.read(os.path.join('test_wavs', 'mix.wav'), dtype='float32')
assert fs == 16000

## inference
# 使用短时傅里叶变换(STFT)将音频转换到频域
input = torch.stft(torch.from_numpy(mix), 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
print("input shape:", input.shape)
# 使用模型进行推理
with torch.no_grad():
    output = model(input[None])[0]
print("output shape:", output.shape)
# 将输出转回复数形式
output_complex = torch.complex(output[..., 0], output[..., 1])
# 使用短时傅里叶逆变换(ISTFT)将频域信号转换回时域
enh = torch.istft(output_complex, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
print("enh shape:", enh.shape)
## save enhanced wav
sf.write(os.path.join('test_wavs', 'enh.wav'), enh.detach().cpu().numpy(), fs)
