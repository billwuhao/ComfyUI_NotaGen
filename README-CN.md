[中文](README-CN.md) | [English](README.md)

# 符号音乐生成. NotaGen 的 ComfyUI 节点.

https://github.com/user-attachments/assets/0671657f-e66b-4000-a0aa-48520f15b782

![image](https://github.com/billwuhao/ComfyUI_NotaGen/blob/master/images/2025-03-10_06-24-03.png)


## 📣 更新

[2025-04-09]⚒️: 不再需要输入 MuseScore4 或 mscore 以及 python 路径, 只需要将 MuseScore4 或 mscore 安装目录例如 `C:\Program Files\MuseScore 4\bin` 添加到系统 path 环境变量即可.

[2025-03-21]⚒️: 增加更多可调参数, 更自由畅玩. 可选是否卸载模型.

[2025-03-15]⚒️: 支持 Linux Ubuntu/Debian 系列, 以及服务器, 其他未测试.
 
本地 Linux 电脑, 安装 `musescore` 等:
```
sudo apt update
sudo apt install musescore
sudo apt install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
```

服务器, 安装虚拟显示器 Xvfb, 其他操作同上:
```
sudo apt update
sudo apt install xvfb
```

[2025-03-13]⚒️: 

- 生成 `.abc` 自动转 `.xml`, `.mp3`, `.png` 格式, 可以听生成的音乐, 同时可以看曲谱啦🎵🎵🎵

- 支持自定义 prompt, 格式必须保持 `<period>|<composer>|<instrumentation>` 的格式, `period`, `composer`, `instrumentation` 的顺序不能乱, 而且以 `|` 分割.

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_NotaGen.git
cd ComfyUI_NotaGen
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

将模型下载放到 `ComfyUI\models\TTS\NotaGen` 下, 并按要求重命名:

[NotaGen-X](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth) → `notagenx.pth`  
[NotaGen-small](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth) → `notagen_small.pth`   
[NotaGen-medium](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth) → `notagen_medium.pth`  
[NotaGen-large](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_0.0001_batch_4.pth) → `notagen_large.pth`  


https://github.com/user-attachments/assets/229139bd-1065-4539-bcfa-b0c245259f6d

## 鸣谢

[NotaGen](https://github.com/ElectricAlexis/NotaGen)