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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nota_model_path = nota_model_path
    node_dir = node_dir
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "model": (s.model_names, {"default": "notagenx.pth"}),
                "period": (s.periods, {"default": "Romantic", }),
                "composer": (s.composers, {"default": "Bach, Johann Sebastian", }),
                "instrumentation": (s.instrumentations, {"default": "Keyboard", }),
                "num_samples": ("INT", {"default": 1, "min": 1}),
                # "temperature": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.1}),
                # "top_k": ("INT", {"default": 50, "min": 0}),
                # "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "abc2xml": ("BOOLEAN", {"default": False}),
                "python_path": ("STRING", {"default": "", "multiline": False, "tooltip": "The absolute path of python.exe"}),
                # "save_path": ("STRING", {"default": "", "tooltip": "(optional) Default Save to output/notagen_xxx"}),
                # "custom_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "(optional) The format must be `period | composer | instrumentation`."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference_patch"
    CATEGORY = "MW-NotaGen"

    # Note_list = Note_list + ['z', 'x']

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

    def inference_patch(self, model, period, composer, instrumentation, num_samples, abc2xml, python_path, seed):
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

        nota_model_path = os.path.join(self.nota_model_path, model)
        checkpoint = torch.load(nota_model_path, map_location=torch.device(self.device))
        nota_model.load_state_dict(checkpoint['model'])
        nota_model = nota_model.to(self.device)
        nota_model.eval()

        prompt_lines=[
            '%' + period + '\n',
            '%' + composer + '\n',
            '%' + instrumentation + '\n']

        patchilizer = Patchilizer(model)

        file_no = 1
        bos_patch = [patchilizer.bos_token_id] * (patch_size - 1) + [patchilizer.eos_token_id]

        while file_no <= num_samples:

            start_time = time.time()
            # start_time_format = time.strftime("%Y%m%d-%H%M%S")

            prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
            byte_list = list(''.join(prompt_lines))
            print(''.join(byte_list), end='')

            prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (patch_size - len(patch)) for patch
                            in prompt_patches]
            prompt_patches.insert(0, bos_patch)

            input_patches = torch.tensor(prompt_patches, device=self.device).reshape(1, -1)

            failure_flag = False
            end_flag = False
            cut_index = None

            tunebody_flag = False
            while True:
                predicted_patch = nota_model.generate(input_patches.unsqueeze(0),
                                                top_k=9,
                                                top_p=0.9,
                                                temperature=1.2)
                if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                    tunebody_flag = True
                    r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(self.device)
                    temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                    predicted_patch = nota_model.generate(temp_input_patches.unsqueeze(0),
                                                    top_k=9,
                                                    top_p=0.9,
                                                    temperature=1.2)
                    predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
                if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                    end_flag = True
                    break
                next_patch = patchilizer.decode([predicted_patch])

                for char in next_patch:
                    byte_list.append(char)
                    print(char, end='')

                patch_end_flag = False
                for j in range(len(predicted_patch)):
                    if patch_end_flag:
                        predicted_patch[j] = patchilizer.special_token_id
                    if predicted_patch[j] == patchilizer.eos_token_id:
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
                    input_patches = patchilizer.encode_generate(abc_code_slice)

                    input_patches = [item for sublist in input_patches for item in sublist]
                    input_patches = torch.tensor([input_patches], device=self.device)
                    input_patches = input_patches.reshape(1, -1)

            if not failure_flag:
                generation_time_cost = time.time() - start_time

                abc_text = ''.join(byte_list)
                filename = time.strftime("%Y%m%d-%H%M%S") + \
                        "_" + format(generation_time_cost, '.2f') + '_' + str(file_no) + ".abc"
                        
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

                    if abc2xml:
                        import subprocess 
                        xml_filename = f"{INTERLEAVED_OUTPUT_FOLDER}/{filename.rsplit(".", 1)[0]}.xml"
                        try:
                            subprocess.run(
                                [python_path, f"{self.node_dir}/abc2xml.py", '-o', INTERLEAVED_OUTPUT_FOLDER, unreduced_output_path, ],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            print(f"Conversion to {xml_filename}",)
                        except subprocess.CalledProcessError as e:
                            print(f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error")
                            raise 
                        
                except:
                    pass
                else:
                    # original
                    original_output_path = os.path.join(ORIGINAL_OUTPUT_FOLDER, filename)
                    with open(original_output_path, 'w') as w:
                        w.write(abc_text)
                        print(f"Saved to {original_output_path}",)
                    
                    if abc2xml:
                        import subprocess 
                        xml_filename = f"{ORIGINAL_OUTPUT_FOLDER}/{filename.rsplit(".", 1)[0]}.xml"
                        try:
                            subprocess.run(
                                [python_path, f"{self.node_dir}/abc2xml.py", '-o', ORIGINAL_OUTPUT_FOLDER, original_output_path, ],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            print(f"Conversion to {xml_filename}",)
                        except subprocess.CalledProcessError as e:
                            print(f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error")
                            raise 
                    file_no += 1

            else:
                print('Generation failed.')

        return (f"Saved to {INTERLEAVED_OUTPUT_FOLDER} and {ORIGINAL_OUTPUT_FOLDER}",)
    

NODE_CLASS_MAPPINGS = {
    "NotaGenRun": NotaGenRun,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NotaGenRun": "NotaGen Run",
}