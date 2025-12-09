import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer

torch.manual_seed(42)


class ResCnnBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        )

        self.last_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.last_relu(x + self.block(x))
    

class CnnDecoder():
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 # rnn_dropout = 0.0,
                 conv_hidden_dim = 16,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 0,
                 patch_stride = 0):
        
        super().__init__()

        self.neural_dim = neural_dim
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign() # basically a shallower tanh 

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        self.input_size = self.neural_dim

        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.tower = nn.Sequential(
            nn.Conv2d(8 * patch_size, conv_hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.1),

            *[ResCnnBlock(conv_hidden_dim) for _ in range(n_layers)],
            nn.Flatten()
        )

        self.out = nn.Linear(64 * conv_hidden_dim, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx):
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        print(x.shape)


import os
    
if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config_path = os.path.join(script_dir, 'rnn_args.yaml')
    
    args = OmegaConf.load(config_path)

    
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()
