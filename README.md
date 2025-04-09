[中文](README-CN.md) | [English](README.md)

# Symbolic Music Generation, NotaGen node for ComfyUI.

https://github.com/user-attachments/assets/0671657f-e66b-4000-a0aa-48520f15b782

![image](https://github.com/billwuhao/ComfyUI_NotaGen/blob/master/images/2025-03-10_06-24-03.png)

## 📣 Updates

[2025-04-09]⚒️: It is no longer necessary to input MuseScore4 or mscore and the Python path. You only need to add the MuseScore4 or mscore installation directory (e.g., `C:\Program Files\MuseScore 4\bin`) to the system path environment variable.

[2025-03-21] ⚒️: Added more tunable parameters for more creative freedom. Optional model unloading.

[2025-03-15]⚒️: Supports Linux Ubuntu/Debian series, as well as servers, others untested, as well as servers.

For local Linux computers, install `musescore` etc.:
```
sudo apt update
sudo apt install musescore
sudo apt install libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xkb1 libxkbcommon-x11-0
```

For servers, install the virtual display Xvfb, other operations are the same as above:
```
sudo apt update
sudo apt install xvfb
```

[2025-03-13]⚒️:

- Automatically convert generated `.abc` to `.xml`, `.mp3`, and `.png` formats.  Now you can listen to the generated music and see the sheet music too! 🎵🎵🎵

- Supports custom prompts. The format must be maintained as `<period>|<composer>|<instrumentation>`, with the order of `period`, `composer`, and `instrumentation` strictly enforced and separated by `|`.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_NotaGen.git
cd ComfyUI_NotaGen
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

Download the model to `ComfyUI\models\TTS\NotaGen` and rename it as required:

[NotaGen-X](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth) → `notagenx.pth`  
[NotaGen-small](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth) → `notagen_small.pth`   
[NotaGen-medium](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth) → `notagen_medium.pth`  
[NotaGen-large](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_0.0001_batch_4.pth) → `notagen_large.pth`  


https://github.com/user-attachments/assets/229139bd-1065-4539-bcfa-b0c245259f6d


## Acknowledgments

[NotaGen](https://github.com/ElectricAlexis/NotaGen)
