[中文](README.md) | [English](README-en.md)

# Symbolic Music Generation, NotaGen node for ComfyUI.

![image](https://github.com/billwuhao/ComfyUI_NotaGen/blob/master/images/2025-03-10_06-24-03.png)


## 📣 Updates

[2025-03-13]⚒️:

- Automatically convert generated `.abc` to `.xml`, `.mp3`, and `.png` formats.  Now you can listen to the generated music and see the sheet music too! 🎵🎵🎵

- Supports custom prompts. The format must be maintained as `<period>|<composer>|<instrumentation>`, with the order of `period`, `composer`, and `instrumentation` strictly enforced and separated by `|`.

- To avoid the hassle of configuring environment variables, please install [MuseScore4](https://musescore.org/en/download) and enter the absolute path of `MuseScore4.exe` into the node, such as `D:\APP\MuseScorePortable\App\MuseScore\bin\MuseScore4.exe`, as well as the absolute path of `python.exe` in ComfyUI, such as `D:\AIGC\APP\ComfyUI_v1\python_embeded\python.exe`.

## Model Download

Download the model to `ComfyUI\models\TTS\NotaGen` and rename it as required:

[NotaGen-X](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth) → `notagenx.pth`  
[NotaGen-small](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth) → `notagen_small.pth`   
[NotaGen-medium](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth) → `notagen_medium.pth`  
[NotaGen-large](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_0.0001_batch_4.pth) → `notagen_large.pth`  


https://github.com/user-attachments/assets/229139bd-1065-4539-bcfa-b0c245259f6d


## Acknowledgments

[NotaGen](https://github.com/ElectricAlexis/NotaGen)