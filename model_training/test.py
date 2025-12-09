import torch
from torch.utils.data import DataLoader
from dataset import BrainToTextDataset, train_test_split_indicies
from omegaconf import OmegaConf
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
    
config_path = os.path.join(script_dir, 'rnn_args.yaml')

args = OmegaConf.load(config_path)

dataset_dir = args['dataset']['dataset_dir']
sessions = args['dataset']['sessions']

train_file_paths = [os.path.join(dataset_dir, s, 'data_train.hdf5') for s in sessions]

train_trials, _ = train_test_split_indicies(train_file_paths, test_percentage=0)

dataset = BrainToTextDataset(
    trial_indicies=train_trials,
    n_batches=10,
    split='train',
    batch_size=4,
    days_per_batch=1
)

loader = DataLoader(dataset, batch_size=None)

batch = next(iter(loader))
print(f"Input Shape: {batch['input_features'].shape}")
print(f"Target Shape: {batch['seq_class_ids'].shape}")
print(f"Time Steps: {batch['n_time_steps']}")
print(batch['input_features'][0].shape)