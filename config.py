# Configurations for model
nota_lx = {
"PATCH_STREAM": True,                                             # Stream training / inference
"PATCH_SIZE": 16,                                                # Patch Size
"PATCH_LENGTH": 1024,                                             # Patch Length
"CHAR_NUM_LAYERS": 6,                                             # Number of layers in the decoder
"PATCH_NUM_LAYERS": 20,                                           # Number of layers in the encoder
"HIDDEN_SIZE": 1280,                                               # Hidden Size
"PATCH_SAMPLING_BATCH_SIZE": 0                                   # Batch size for patch during training, 0 for full conaudio
}

nota_small = {
"PATCH_STREAM": True,                                             # Stream training / inference
"PATCH_SIZE": 16,                                                # Patch Size
"PATCH_LENGTH": 2048,                                             # Patch Length
"CHAR_NUM_LAYERS": 3,                                             # Number of layers in the decoder
"PATCH_NUM_LAYERS": 12,                                           # Number of layers in the encoder
"HIDDEN_SIZE": 768,                                               # Hidden Size
"PATCH_SAMPLING_BATCH_SIZE": 0                                   # Batch size for patch during training, 0 for full conaudio
}

nota_medium = {
"PATCH_STREAM": True,                                             # Stream training / inference
"PATCH_SIZE": 16,                                                # Patch Size
"PATCH_LENGTH": 2048,                                             # Patch Length
"CHAR_NUM_LAYERS": 3,                                             # Number of layers in the decoder
"PATCH_NUM_LAYERS": 16,                                           # Number of layers in the encoder
"HIDDEN_SIZE": 1024,                                               # Hidden Size
"PATCH_SAMPLING_BATCH_SIZE": 0                                   # Batch size for patch during training, 0 for full conaudio
}