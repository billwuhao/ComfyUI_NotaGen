import os
import time
import torch
from .utils import *
from .config import nota_lx, nota_small, nota_medium
from transformers import GPT2Config
from abctoolkit.utils import Barline_regexPattern
# from abctoolkit.transpose import Note_list, Pitch_sign_list
from abctoolkit.duration import calculate_bartext_duration

node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(node_dir))
output_path = os.path.join(comfy_path, "output")
# Path to weights for inference
nota_model_path = os.path.join(comfy_path, "models", "TTS", "NotaGen")

# Folder to save output files
ORIGINAL_OUTPUT_FOLDER = os.path.join(output_path, 'notagen_original')
INTERLEAVED_OUTPUT_FOLDER = os.path.join(output_path, 'notagen_interleaved')

os.makedirs(ORIGINAL_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INTERLEAVED_OUTPUT_FOLDER, exist_ok=True)


MODEL_CACHE = None
PATCHILIZER = None
class NotaGenRun:
    model_names = ["notagenx.pth", "notagen_small.pth", "notagen_medium.pth", "notagen_large.pth"]
    periods = ["Baroque", "Classical", "Romantic"]
    composers = ["Bach, Johann Sebastian", "Corelli, Arcangelo", "Handel, George Frideric", "Scarlatti, Domenico", "Vivaldi, Antonio", "Beethoven, Ludwig van", 
                "Haydn, Joseph", "Mozart, Wolfgang Amadeus", "Paradis, Maria Theresia von", "Reichardt, Louise", "Saint-Georges, Joseph Bologne", "Schroter, Corona", 
                "Bartok, Bela", "Berlioz, Hector", "Bizet, Georges", "Boulanger, Lili", "Boulton, Harold", "Brahms, Johannes", "Burgmuller, Friedrich", 
                "Butterworth, George", "Chaminade, Cecile", "Chausson, Ernest", "Chopin, Frederic", "Cornelius, Peter", "Debussy, Claude", "Dvorak, Antonin", 
                "Faisst, Clara", "Faure, Gabriel", "Franz, Robert", "Gonzaga, Chiquinha", "Grandval, Clemence de", "Grieg, Edvard", "Hensel, Fanny", 
                "Holmes, Augusta Mary Anne", "Jaell, Marie", "Kinkel, Johanna", "Kralik, Mathilde", "Lang, Josephine", "Lehmann, Liza", "Liszt, Franz", 
                "Mayer, Emilie", "Medtner, Nikolay", "Mendelssohn, Felix", "Munktell, Helena", "Parratt, Walter", "Prokofiev, Sergey", "Rachmaninoff, Sergei", 
                "Ravel, Maurice", "Saint-Saens, Camille", "Satie, Erik", "Schubert, Franz", "Schumann, Clara", "Schumann, Robert", "Scriabin, Aleksandr", 
                "Shostakovich, Dmitry", "Sibelius, Jean", "Smetana, Bedrich", "Tchaikovsky, Pyotr", "Viardot, Pauline", "Warlock, Peter", "Wolf, Hugo", "Zumsteeg, Emilie"]
    instrumentations = ["Chamber", "Choral", "Keyboard", "Orchestral", "Vocal-Orchestral", "Art Song"]
    
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        nota_model_path = nota_model_path
        self.node_dir = node_dir

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (s.model_names, {"default": "notagenx.pth"}),
                "period": (s.periods, {"default": "Romantic"}),
                "composer": (s.composers, {"default": "Bach, Johann Sebastian"}),
                "instrumentation": (s.instrumentations, {"default": "Keyboard"}),
                "custom_prompt": ("STRING", {
                    "default": "Romantic | Bach, Johann Sebastian | Keyboard", 
                    "multiline": True, 
                    "tooltip": "Custom prompt must follow format: <period>|<composer>|<instrumentation>"
                }),
                "unload_model":("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 5, "step": 0.1}),
                "top_k": ("INT", {"default": 50, "min": 0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("AUDIO", "IMAGE", "STRING")
    RETURN_NAMES = ("audio", "score", "message")
    FUNCTION = "inference_patch"
    CATEGORY = "🎤MW/MW-NotaGen"

    def inference_patch(self, model, period, composer, instrumentation, 
                        custom_prompt,
                        unload_model,
                        temperature,
                        top_k,
                        top_p,
                        seed):
        if seed != 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        if model == "notagenx.pth" or model == "notagen_large.pth":
            cf = nota_lx
        elif model == "notagen_small.pth":
            cf = nota_small
        elif model == "notagen_medium.pth":
            cf = nota_medium
        patch_size = cf["PATCH_SIZE"]
        patch_length = cf["PATCH_LENGTH"]
        char_num_layers = cf["CHAR_NUM_LAYERS"]
        patch_num_layers = cf["PATCH_NUM_LAYERS"]
        hidden_size = cf["HIDDEN_SIZE"]

        patch_config = GPT2Config(num_hidden_layers=patch_num_layers,
                                max_length=patch_length,
                                max_position_embeddings=patch_length,
                                n_embd=hidden_size,
                                num_attention_heads=hidden_size // 64,
                                vocab_size=1)
        byte_config = GPT2Config(num_hidden_layers=char_num_layers,
                                max_length=patch_size + 1,
                                max_position_embeddings=patch_size + 1,
                                hidden_size=hidden_size,
                                num_attention_heads=hidden_size // 64,
                                vocab_size=128)

        nota_model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config, model=model)

        print("Parameter Number: " + str(sum(p.numel() for p in nota_model.parameters() if p.requires_grad)))

        nota_model_path = os.path.join(nota_model_path, model)

        global MODEL_CACHE, PATCHILIZER
        if MODEL_CACHE is None:
            MODEL_CACHE = torch.load(nota_model_path, map_location=torch.device(self.device))
            nota_model.load_state_dict(MODEL_CACHE['model'])
            nota_model = nota_model.to(self.device)
            nota_model.eval()

        if custom_prompt.strip():
            period, composer, instrumentation = [i.strip() for i in custom_prompt.split('|')]

        prompt_lines=[
            '%' + period + '\n',
            '%' + composer + '\n',
            '%' + instrumentation + '\n']

        if PATCHILIZER is None:
            PATCHILIZER  = Patchilizer(model)

        # file_no = 1
        bos_patch = [PATCHILIZER.bos_token_id] * (patch_size - 1) + [PATCHILIZER.eos_token_id]
        num_gen = 0
        unreduced_xml_path = None
        save_xml_original = False
        while num_gen <= 5: #num_samples:

            start_time = time.time()
            # start_time_format = time.strftime("%Y%m%d-%H%M%S")

            prompt_patches = PATCHILIZER.patchilize_metadata(prompt_lines)
            byte_list = list(''.join(prompt_lines))
            print(''.join(byte_list), end='')

            prompt_patches = [[ord(c) for c in patch] + [PATCHILIZER.special_token_id] * (patch_size - len(patch)) for patch
                            in prompt_patches]
            prompt_patches.insert(0, bos_patch)

            input_patches = torch.tensor(prompt_patches, device=self.device).reshape(1, -1)

            failure_flag = False
            end_flag = False
            cut_index = None

            tunebody_flag = False
            while True:
                predicted_patch = nota_model.generate(input_patches.unsqueeze(0),
                                                top_k=top_k,
                                                top_p=top_p,
                                                temperature=temperature)
                if not tunebody_flag and PATCHILIZER.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                    tunebody_flag = True
                    r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(self.device)
                    temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                    predicted_patch = nota_model.generate(temp_input_patches.unsqueeze(0),
                                                    top_k=top_k,
                                                    top_p=top_p,
                                                    temperature=temperature)
                    predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
                if predicted_patch[0] == PATCHILIZER.bos_token_id and predicted_patch[1] == PATCHILIZER.eos_token_id:
                    end_flag = True
                    break
                next_patch = PATCHILIZER.decode([predicted_patch])

                for char in next_patch:
                    byte_list.append(char)
                    print(char, end='')

                patch_end_flag = False
                for j in range(len(predicted_patch)):
                    if patch_end_flag:
                        predicted_patch[j] = PATCHILIZER.special_token_id
                    if predicted_patch[j] == PATCHILIZER.eos_token_id:
                        patch_end_flag = True

                predicted_patch = torch.tensor([predicted_patch], device=self.device)  # (1, 16)
                input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

                if len(byte_list) > 102400:  
                    failure_flag = True
                    break
                if time.time() - start_time > 20 * 60:  
                    failure_flag = True
                    break

                if input_patches.shape[1] >= patch_length * patch_size and not end_flag:
                    print('Stream generating...')
                    abc_code = ''.join(byte_list)
                    abc_lines = abc_code.split('\n')

                    tunebody_index = None
                    for i, line in enumerate(abc_lines):
                        if line.startswith('[r:') or line.startswith('[V:'):
                            tunebody_index = i
                            break
                    if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                        break

                    metadata_lines = abc_lines[:tunebody_index]
                    tunebody_lines = abc_lines[tunebody_index:]

                    metadata_lines = [line + '\n' for line in metadata_lines]
                    if not abc_code.endswith('\n'):  
                        tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [
                            tunebody_lines[-1]]
                    else:
                        tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]

                    if cut_index is None:
                        cut_index = len(tunebody_lines) // 2

                    abc_code_slice = ''.join(metadata_lines + tunebody_lines[-cut_index:])
                    input_patches = PATCHILIZER.encode_generate(abc_code_slice)

                    input_patches = [item for sublist in input_patches for item in sublist]
                    input_patches = torch.tensor([input_patches], device=self.device)
                    input_patches = input_patches.reshape(1, -1)

            if not failure_flag:
                generation_time_cost = time.time() - start_time

                abc_text = ''.join(byte_list)
                filename = time.strftime("%Y%m%d-%H%M%S") + \
                        "_" + str(int(generation_time_cost)) + ".abc"
                        
                # unreduce
                unreduced_output_path = os.path.join(INTERLEAVED_OUTPUT_FOLDER, filename)
                
                abc_lines = abc_text.split('\n')
                abc_lines = list(filter(None, abc_lines))
                abc_lines = [line + '\n' for line in abc_lines]
                
                try:
                    abc_lines = self.rest_unreduce(abc_lines)

                    with open(unreduced_output_path, 'w') as file:
                        file.writelines(abc_lines)
                        print(f"Saved to {unreduced_output_path}",)
                    unreduced_xml_path = self.convert_abc2xml(unreduced_output_path, INTERLEAVED_OUTPUT_FOLDER)
                    if unreduced_xml_path:
                        save_xml_original = True
                    else:
                        print("Conversion xml failed.")
                        num_gen += 1
                        save_xml_original = False
                        
                except:
                    num_gen += 1
                    continue
                else:
                    # original
                    original_output_path = os.path.join(ORIGINAL_OUTPUT_FOLDER, filename)
                    with open(original_output_path, 'w') as w:
                        w.write(abc_text)
                        print(f"Saved to {original_output_path}",)
                    
                    if save_xml_original:
                        original_xml_path = self.convert_abc2xml(original_output_path, ORIGINAL_OUTPUT_FOLDER)
                        if original_xml_path:
                            print(f"Conversion to {original_xml_path}",)
                        break
                    else:
                        num_gen += 1
                        continue
                    # file_no += 1
            else:
                print('Generation failed.')
                num_gen += 1
                if num_gen > 5:
                    raise Exception("All generation attempts failed after 6 tries. Try again.")

        if unreduced_xml_path:
            mp3_path = self.xml2mp3(unreduced_xml_path)
            png_paths = self.xml2png(unreduced_xml_path)
                
            # 处理音频
            audio = None
            if mp3_path and os.path.exists(mp3_path):
                import torchaudio
                waveform, sample_rate = torchaudio.load(mp3_path)
                audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            else:
                audio = self.get_empty_audio()
            
            # 处理图片
            images = []
            if png_paths:
                from PIL import Image, ImageOps
                import numpy as np
                
                for image_path in png_paths:
                    i = Image.open(image_path)
                    # 创建一个白色背景的图像
                    image = Image.new("RGB", i.size, (255, 255, 255))

                    # 将透明背景的图片粘贴到白色背景上
                    image.paste(i, mask=i.split()[3])  # 使用 Alpha 通道作为掩码
                    # i = ImageOps.exif_transpose(i) # 翻转图片
                    
                    # 调整宽度为1024，保持宽高比
                    # width, height = i.size
                    # new_width = 1024
                    # new_height = int(height * (new_width / width))
                    # i = i.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    image = image.convert("RGB")
                    image = np.array(image).astype(np.float32) / 255.0
                    image = torch.from_numpy(image)[None,]
                    images.append(image)
                
                if len(images) > 1:
                    image1 = images[0]
                    for image2 in images[1:]:
                        image1 = torch.cat((image1, image2), dim=0)
                else:
                    image1 = images[0]
            else:
                image1 = self.get_empty_image()

            if unload_model:
                import gc
                PATCHILIZER = None
                nota_model = None
                MODEL_CACHE = None
                gc.collect()
                torch.cuda.empty_cache()

            return (
                audio,
                image1,
                f"Saved to {INTERLEAVED_OUTPUT_FOLDER} and {ORIGINAL_OUTPUT_FOLDER}",
            )
        
        else:
            if unload_model:
                import gc
                PATCHILIZER = None
                nota_model = None
                MODEL_CACHE = None
                gc.collect()
                torch.cuda.empty_cache()
                
            print(f".abc and .xml was saved to {INTERLEAVED_OUTPUT_FOLDER} and {ORIGINAL_OUTPUT_FOLDER}")
            raise Exception("Conversion of .mp3 and .png failed, try again or check if MuseScore4 installation was successful.")


    def get_empty_audio(self):
        """Return empty audio"""
        return {"waveform": torch.zeros(1, 2, 1), "sample_rate": 44100}

    def get_empty_image(self):
        """Return empty image"""
        import numpy as np
        return torch.from_numpy(np.zeros((1, 64, 64, 3), dtype=np.float32))

    def rest_unreduce(self, abc_lines):

        tunebody_index = None
        for i in range(len(abc_lines)):
            if '[V:' in abc_lines[i]:
                tunebody_index = i
                break

        metadata_lines = abc_lines[: tunebody_index]
        tunebody_lines = abc_lines[tunebody_index:]

        part_symbol_list = []
        voice_group_list = []
        for line in metadata_lines:
            if line.startswith('%%score'):
                for round_bracket_match in re.findall(r'\((.*?)\)', line):
                    voice_group_list.append(round_bracket_match.split())
                existed_voices = [item for sublist in voice_group_list for item in sublist]
            if line.startswith('V:'):
                symbol = line.split()[0]
                part_symbol_list.append(symbol)
                if symbol[2:] not in existed_voices:
                    voice_group_list.append([symbol[2:]])
        z_symbol_list = []  # voices that use z as rest
        x_symbol_list = []  # voices that use x as rest
        for voice_group in voice_group_list:
            z_symbol_list.append('V:' + voice_group[0])
            for j in range(1, len(voice_group)):
                x_symbol_list.append('V:' + voice_group[j])

        part_symbol_list.sort(key=lambda x: int(x[2:]))

        unreduced_tunebody_lines = []

        for i, line in enumerate(tunebody_lines):
            unreduced_line = ''

            line = re.sub(r'^\[r:[^\]]*\]', '', line)

            pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
            matches = re.findall(pattern, line)

            line_bar_dict = {}
            for match in matches:
                key = f'V:{match[0]}'
                value = match[1]
                line_bar_dict[key] = value

            # calculate duration and collect barline
            dur_dict = {}  
            for symbol, bartext in line_bar_dict.items():
                right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
                bartext = bartext[:-len(right_barline)]
                try:
                    bar_dur = calculate_bartext_duration(bartext)
                except:
                    bar_dur = None
                if bar_dur is not None:
                    if bar_dur not in dur_dict.keys():
                        dur_dict[bar_dur] = 1
                    else:
                        dur_dict[bar_dur] += 1

            try:
                ref_dur = max(dur_dict, key=dur_dict.get)
            except:
                pass    # use last ref_dur

            if i == 0:
                prefix_left_barline = line.split('[V:')[0]
            else:
                prefix_left_barline = ''

            for symbol in part_symbol_list:
                if symbol in line_bar_dict.keys():
                    symbol_bartext = line_bar_dict[symbol]
                else:
                    if symbol in z_symbol_list:
                        symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + right_barline
                    elif symbol in x_symbol_list:
                        symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + right_barline
                unreduced_line += '[' + symbol + ']' + symbol_bartext

            unreduced_tunebody_lines.append(unreduced_line + '\n')

        unreduced_lines = metadata_lines + unreduced_tunebody_lines

        return unreduced_lines

    def wait_for_file(self, file_path, timeout=15, check_interval=0.3):
        """Wait for file generation to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(file_path):
                # 对于MP3文件，检查文件大小是否不再变化
                if file_path.endswith('.mp3'):
                    initial_size = os.path.getsize(file_path)
                    time.sleep(check_interval)
                    if os.path.getsize(file_path) == initial_size:
                        return True
                else:
                    return True
            time.sleep(check_interval)
        return False

    def wait_for_png_sequence(self, base_path, timeout=15, check_interval=0.3):
        """Wait for PNG sequence generation to complete"""
        import glob
        
        start_time = time.time()
        last_count = 0
        stable_count = 0
        
        while time.time() - start_time < timeout:
            current_files = glob.glob(f"{base_path}-*.png")
            current_count = len(current_files)
            
            if current_count > 0:
                if current_count == last_count:
                    stable_count += 1
                    if stable_count >= 3:  # 连续3次检查文件数量不变
                        return sorted(current_files)
                else:
                    stable_count = 0
            
            last_count = current_count
            time.sleep(check_interval)
        
        return None

    def xml2mp3(self, xml_path):
        import subprocess
        import sys
        import tempfile

        mp3_path = xml_path.rsplit(".", 1)[0] + ".mp3"
        # 检测操作系统是否为 Linux
        if sys.platform == "linux":
            try:
                # 使用不同的显示端口
                display_number = 100
                os.environ["DISPLAY"] = f":{display_number}"

                # 检查并清理旧的 Xvfb 锁文件
                tmp_dir = tempfile.mkdtemp()
                xvfb_lock_file = os.path.join(tmp_dir, f".X{display_number}-lock")
                if os.path.exists(xvfb_lock_file):
                    print(f"清理旧的 Xvfb 锁文件: {xvfb_lock_file}")
                    os.remove(xvfb_lock_file)

                # 杀死所有残留的 Xvfb 进程
                subprocess.run(["pkill", "Xvfb"], stderr=subprocess.DEVNULL)  # 忽略错误
                time.sleep(1)  # 等待进程终止

                # 启动 Xvfb
                xvfb_process = subprocess.Popen(["Xvfb", f":{display_number}", "-screen", "0", "1024x768x24"])
                time.sleep(2)  # 等待 Xvfb 启动

                # 设置 Qt 插件环境变量
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
                
                # 运行 mscore 命令
                subprocess.run(
                    ['mscore', '-o', mp3_path, xml_path],
                    check=True,
                    capture_output=True,
                )
                
                # 等待MP3文件生成完成
                if self.wait_for_file(mp3_path):
                    print(f"Conversion to {mp3_path} completed")
                    return mp3_path
                else:
                    print("MP3 conversion timeout")
                    return None
            except subprocess.CalledProcessError as e:
                print(f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error")
                return None
            finally:
                # 关闭 Xvfb
                xvfb_process.terminate()
                xvfb_process.wait()
        else:
            try:
                import shutil
                musescore_executable_path = shutil.which('MuseScore4')
                print(musescore_executable_path)
                subprocess.run(
                    [musescore_executable_path, '-o', mp3_path, xml_path],
                    check=True,
                    capture_output=True,
                )
                # 等待MP3文件生成完成
                if self.wait_for_file(mp3_path):
                    print(f"Conversion to {mp3_path} completed")
                    return mp3_path
                else:
                    print("MP3 conversion timeout")
                    return None
            except subprocess.CalledProcessError as e:
                print(f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error")
                return None

    def xml2png(self, xml_path):
        import subprocess
        import sys
        import tempfile
        
        base_png_path = xml_path.rsplit(".", 1)[0]
        # 检测操作系统是否为 Linux
        if sys.platform == "linux":
            try:
                # 使用不同的显示端口
                display_number = 100
                os.environ["DISPLAY"] = f":{display_number}"

                # 检查并清理旧的 Xvfb 锁文件
                tmp_dir = tempfile.mkdtemp()
                xvfb_lock_file = os.path.join(tmp_dir, f".X{display_number}-lock")
                if os.path.exists(xvfb_lock_file):
                    print(f"清理旧的 Xvfb 锁文件: {xvfb_lock_file}")
                    os.remove(xvfb_lock_file)

                # 杀死所有残留的 Xvfb 进程
                subprocess.run(["pkill", "Xvfb"], stderr=subprocess.DEVNULL)  # 忽略错误
                time.sleep(1)  # 等待进程终止

                # 启动 Xvfb
                xvfb_process = subprocess.Popen(["Xvfb", f":{display_number}", "-screen", "0", "1024x768x24"])
                time.sleep(2)  # 等待 Xvfb 启动

                # 设置 Qt 插件环境变量
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
                
                # 运行 mscore 命令
                subprocess.run(
                    ['mscore', '-o', f"{base_png_path}.png", xml_path],
                    check=True,
                    capture_output=True,
                )
                # 等待PNG序列生成完成
                png_files = self.wait_for_png_sequence(base_png_path)
                if png_files:
                    print(f"Converted to {len(png_files)} PNG files")
                    return png_files
                else:
                    print("PNG conversion timeout")
                    return None
            except subprocess.CalledProcessError as e:
                print(f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error")
                return None
            finally:
                # 关闭 Xvfb
                xvfb_process.terminate()
                xvfb_process.wait()
        else:
            try:
                import shutil
                musescore_executable_path = shutil.which('MuseScore4')
                print(musescore_executable_path)
                subprocess.run(
                    [musescore_executable_path, '-o', f"{base_png_path}.png", xml_path],
                    check=True,
                    capture_output=True,
                )
                # 等待PNG序列生成完成
                png_files = self.wait_for_png_sequence(base_png_path)
                if png_files:
                    print(f"Converted to {len(png_files)} PNG files")
                    return png_files
                else:
                    print("PNG conversion timeout")
                    return None
            except subprocess.CalledProcessError as e:
                print(f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error")
                return None

    def convert_abc2xml(self, abc_path, output_dir):
        import sys
        import os
        sys.path.append(self.node_dir)
        from abc2xml import getXmlDocs, writefile, readfile, info
        xml_path = abc_path.rsplit(".", 1)[0] + ".xml"
        try:
            fnm, ext = os.path.splitext(abc_path)
            abctext = readfile(abc_path)
            
            # 设置参数，对应原命令行参数
            skip, num = 0, 1  # 对应 -m 参数
            show_whole_rests = False  # 对应 -r 参数
            line_breaks = False  # 对应 -b 参数
            force_string_fret = False  # 对应 -f 参数
            
            xml_docs = getXmlDocs(abctext, skip, num, show_whole_rests, line_breaks, force_string_fret)
            
            for itune, xmldoc in enumerate(xml_docs):
                fnmNum = '%02d' % (itune + 1) if len(xml_docs) > 1 else ''
                writefile(output_dir, fnm, fnmNum, xmldoc, '', False)  # '' 对应 -mxl 参数，False 对应 -t 参数
            print(f"Conversion to {xml_path}",)
            return xml_path
        except Exception as e:
            print(f"Conversion failed: {str(e)}")
            return None


NODE_CLASS_MAPPINGS = {
    "NotaGenRun": NotaGenRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NotaGenRun": "NotaGen Run",
}