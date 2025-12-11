import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

# Ensure these files are in the same directory
from rnn_model import GRUDecoder
from my_train import CnnDecoder, CnnGruDecoder, CnnTransformerDecoder
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep, LOGIT_TO_PHONEME

def main():
    class Args:
        # CHANGE THIS: Remove '/checkpoint/best_checkpoint' from the end
        model_path = 'C:/Users/3verj/OneDrive/Desktop/nejm-brain-to-text-2/trained_models/best_pure_cnn'  
        
        # POINT THIS to where your HDF5 data lives
        data_dir = 'C:/Users/3verj/OneDrive/Desktop/nejm-brain-to-text-2/data/hdf5_data_final' 
        
        # Set to 'val' to see Error Rates, or 'test' for just predictions
        eval_type = 'val' 
        
        # Path to your metadata CSV
        csv_path = 'C:/Users/3verj/OneDrive/Desktop/nejm-brain-to-text-2/data/t15_copyTaskData_description.csv'
        
        gpu_number = 0

    args = Args()

    # -------------------------------------------------------------------------
    # 2. Setup Device and Configuration
    # -------------------------------------------------------------------------
    model_path = args.model_path
    data_dir = args.data_dir
    eval_type = args.eval_type

    # Load metadata and model config
    b2txt_csv_df = pd.read_csv(args.csv_path)
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

    # Setup GPU
    gpu_number = args.gpu_number
    if torch.cuda.is_available() and gpu_number >= 0:
        device = torch.device(f'cuda:{gpu_number}')
        print(f'Using {device} for inference.')
    else:
        print('Using CPU for inference.')
        device = torch.device('cpu')

    # -------------------------------------------------------------------------
    # 3. Initialize and Load Model
    # -------------------------------------------------------------------------
    
    
    print("Loading model...")
    model = CnnDecoder(
        neural_dim = model_args['model']['n_input_features'],
        n_units = model_args['model']['n_units'], 
        n_days = len(model_args['dataset']['sessions']),
        n_classes = model_args['dataset']['n_classes'],
        rnn_dropout = model_args['model']['rnn_dropout'],
        input_dropout = model_args['model']['input_network']['input_layer_dropout'],
        n_layers = model_args['model']['n_layers'],
        patch_size = model_args['model']['patch_size'],
        patch_stride = model_args['model']['patch_stride'],
    )

    # Load weights
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # Fix keys if trained with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 4. Load Data
    # -------------------------------------------------------------------------
    test_data = {}
    total_trials = 0
    print(f"Loading {eval_type} data...")
    
    for session in model_args['dataset']['sessions']:
        session_dir = os.path.join(data_dir, session)
        if not os.path.isdir(session_dir):
            continue
            
        files = [f for f in os.listdir(session_dir) if f.endswith('.hdf5')]
        if f'data_{eval_type}.hdf5' in files:
            eval_file = os.path.join(session_dir, f'data_{eval_type}.hdf5')
            data = load_h5py_file(eval_file, b2txt_csv_df)
            test_data[session] = data
            total_trials += len(data["neural_features"])
            print(f'  Loaded {len(data["neural_features"])} trials for session {session}.')

    print(f'Total trials to evaluate: {total_trials}\n')

    # -------------------------------------------------------------------------
    # 5. Run Neural Decoding (Inference)
    # -------------------------------------------------------------------------
    print("Running inference...")
    results_list = []
    
    # Metrics for PER calculation
    total_edit_distance = 0
    total_phoneme_length = 0

    with torch.no_grad():
        with tqdm(total=total_trials, unit='trial') as pbar:
            for session, data in test_data.items():
                input_layer = model_args['dataset']['sessions'].index(session)
                
                for i in range(len(data['neural_features'])):
                    # Prepare Input
                    neural_input = data['neural_features'][i]
                    neural_input = np.expand_dims(neural_input, axis=0) # Add batch dim
                    neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)

                    # --- DECODING STEP ---
                    logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
                    
                    # Convert Logits to Phonemes
                    pred_indices = np.argmax(logits[0], axis=-1)
                    
                    # CTC-style decoding: remove blanks (0) and consecutive duplicates
                    decoded_indices = []
                    for k, idx in enumerate(pred_indices):
                        if idx != 0: # not blank
                            if k == 0 or idx != pred_indices[k-1]:
                                decoded_indices.append(idx)
                    
                    pred_phonemes = [LOGIT_TO_PHONEME[p] for p in decoded_indices]
                    pred_seq_str = " ".join(pred_phonemes)

                    # Get Metadata
                    block_num = data['block_num'][i]
                    trial_num = data['trial_num'][i]
                    
                    # --- EVALUATION (PER) ---
                    true_phonemes = []
                    true_seq_str = ""
                    dist = 0
                    
                    if eval_type == 'val':
                        # Retrieve Ground Truth
                        true_ids = data['seq_class_ids'][i][0:data['seq_len'][i]]
                        true_phonemes = [LOGIT_TO_PHONEME[p] for p in true_ids]
                        true_seq_str = " ".join(true_phonemes)
                        
                        # Calculate Distance
                        dist = editdistance.eval(true_phonemes, pred_phonemes)
                        total_edit_distance += dist
                        total_phoneme_length += len(true_phonemes)

                    # Store results
                    results_list.append({
                        'session': session,
                        'block': block_num,
                        'trial': trial_num,
                        'true_phonemes': true_seq_str,
                        'pred_phonemes': pred_seq_str,
                        'edit_distance': dist if eval_type == 'val' else None,
                        'seq_len': len(true_phonemes) if eval_type == 'val' else None
                    })
                    
                    pbar.update(1)

    # -------------------------------------------------------------------------
    # 6. Report Results
    # -------------------------------------------------------------------------
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    # Print first few examples
    for res in results_list[:5]:
        print(f"Session {res['session']} | Block {res['block']} | Trial {res['trial']}")
        if eval_type == 'val':
            print(f"True: {res['true_phonemes']}")
        print(f"Pred: {res['pred_phonemes']}")
        print("-" * 30)

    if eval_type == 'val' and total_phoneme_length > 0:
        per = 100 * total_edit_distance / total_phoneme_length
        print("\n" + "*"*50)
        print(f"FINAL METRICS")
        print("*"*50)
        print(f"Total Edit Distance: {total_edit_distance}")
        print(f"Total Phonemes:      {total_phoneme_length}")
        print(f"Phoneme Error Rate:  {per:.2f}%")
        print("*"*50)
    
    # Save to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(model_path, f'phoneme_preds_{eval_type}_{timestamp}.csv')
    pd.DataFrame(results_list).to_csv(out_file, index=False)
    print(f"\nDetailed predictions saved to: {out_file}")

if __name__ == "__main__":
    main()