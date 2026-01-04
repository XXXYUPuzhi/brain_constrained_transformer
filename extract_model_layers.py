# -*- coding: utf-8 -*-
"""
Created on Wed. Oct. 1 21:45:03 2025

@author: ypz
"""
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from building_blocks import visual2action_3
from attn_eye_correlation_analyses import prepare_data
import utils

def main():
    # --- Configuration ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR = model_paths = './omega_models_fromcluster'
    GAME_DATA_DIR = './junction_predict_resortind'
    EYE_DATA_DIR = './eyecorrection_fromcluster/linear/'
    
    # Specific file targets
    MODEL_FILENAME = '140sess_2layers_focalloss_12d1h_noEmbLN_0768b.pkl'
    EYE_FILE_INDEX = 3  # Index of the eye tracking file to analyze

    # --- 1. Load Model ---
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    print(f"Loading model from {model_path}...")
    
    try:
        with open(model_path, 'rb') as file:
            _, state_dict, configs = pickle.load(file)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    model = visual2action_3(configs).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode

    # --- 2. Prepare Data ---
    fns_board = os.listdir(GAME_DATA_DIR)
    fns_eye = os.listdir(EYE_DATA_DIR)
    
    target_eye_fn = fns_eye[EYE_FILE_INDEX]
    
    # Find corresponding board file based on substring match
    try:
        i_boardfile = np.where([target_eye_fn[:3] in x for x in fns_board])[0][0]
    except IndexError:
        print(f"No matching board file found for {target_eye_fn}")
        return

    print(f"Processing: {target_eye_fn} -> Board Index: {i_boardfile}")

    # Unpacking data loader (ignoring unused variables with _)
    game_info, action, board_inds, observation, _, _, s_, a_ = prepare_data(
        i_boardfile, '', GAME_DATA_DIR, device=DEVICE
    )

    # --- 3. Manual Forward Pass (Layer-wise Extraction) ---
    # We manually step through the transformer blocks to access intermediate outputs.
    
    with torch.no_grad():
        B, H, W, D = s_.shape
        
        # Embedding Layer
        emb = model.to_patch_embedding(s_)
        emb_with_pos = emb + model.input_pos_emb.expand(B, -1, -1)
        emb_with_CLS = torch.cat((model.act_token.expand(B, -1, -1), emb_with_pos), dim=1)

        # Encoder Block 0
        # X = model.encode[0](emb_with_CLS) # Standard pass
        
        # Manual pass Block 0
        emb_LN = model.encode[0].norm1(emb_with_CLS)
        attn1, attn1_score, q1, k1, v1 = model.encode[0].attn(emb_LN)
        attn1_res = attn1 + emb_with_CLS

        attn1_res_LN = model.encode[0].norm2(attn1_res)
        mlp1 = model.encode[0].mlp(attn1_res_LN)
        mlp1_res = attn1_res + mlp1

        # Encoder Block 1
        # Manual pass Block 1
        mlp1_res_LN = model.encode[1].norm1(mlp1_res)
        attn2, attn2_score, q2, k2, v2 = model.encode[1].attn(mlp1_res_LN)
        attn2_res = attn2 + mlp1_res

        attn2_res_LN = model.encode[1].norm2(attn2_res)
        mlp2 = model.encode[1].mlp(attn2_res_LN)
        mlp2_res = attn2_res + mlp2

        # Prediction Head
        # Using the CLS token (index 0) for prediction
        y = model.predhead(mlp2_res[:, 0, :])
        y = F.softmax(y, dim=-1)

    # --- 4. Output Results ---
    print(f"Layer 1 MLP Output Shape: {mlp1.shape}")
    print(f"Layer 2 MLP Output Shape: {mlp2.shape}")
    print(f"Prediction Shape: {y.shape}")

if __name__ == "__main__":
    main()