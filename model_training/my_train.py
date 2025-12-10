import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from omegaconf import OmegaConf

from rnn_model import GRUDecoder

torch.manual_seed(42)



class ResCnnBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.1),  
            nn.Dropout(dropout),

            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
        )

        self.last_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.last_relu(x + self.block(x))
    

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)

        # Apply sin to even indices (2i)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices (2i+1)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Seq_Len, Batch, Dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    

class CnnDecoder(nn.Module):
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout = 0.0,
                 conv_hidden_dim = 512,
                 input_dropout = 0.0,
                 n_layers = 2, 
                 patch_size = 10,
                 patch_stride = 4):
        
        super().__init__()

        self.neural_dim = neural_dim
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days
        self.input_dropout = input_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride


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

        self.tower = nn.Sequential(
            nn.Conv2d(8 * patch_size, conv_hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.1),

            *[ResCnnBlock(conv_hidden_dim, rnn_dropout) for _ in range(n_layers)],
            nn.Flatten()
        )

        self.out = nn.Linear(64 * conv_hidden_dim, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, day_idx):
        # input x shape: [B, T, F]
        B, T, _ = x.shape

        # Set up day specific transformation
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        # Einstien sum on day weights and x tensor
        # [T, F] x [512, 512] -> [B, T, 512]
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.training and self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # [B, T, 8, 8, 8] (8 of 8x8 brain electrode arrays)
        x = x.view(B, T, 8, 8, 8)

        # [B, T, 8, 8, 8, patch_size] "stack" feature maps on time axis using patch size and stride 
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        
        # [B, T, 8, patch_size, 8, 8]
        x = x.permute(0, 1, 2, 5, 3, 4).contiguous()

        num_windows = x.shape[1]

        # [B, T, 8 * patch_size, 8, 8]
        x = x.view(-1, 8 * self.patch_size, 8, 8)

        # [B * T, hidden_dim * 64]
        feats = self.tower(x)

        # [B, T, hidden_dim * 64]
        feats = feats.view(B, num_windows, -1)

        # [B, T, 41]
        logits = self.out(feats)

        return logits


class CnnGruDecoder(nn.Module):
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 cnn_layers = 2,
                 rnn_dropout = 0.0,
                 conv_hidden_dim = 64,
                 input_dropout = 0.0,
                 n_layers = 5, 
                 patch_size = 14,
                 patch_stride = 4):
        
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days
        self.input_dropout = input_dropout
        self.rnn_dropout = rnn_dropout

        self.patch_size = patch_size
        self.patch_stride = patch_stride


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

        self.tower = nn.Sequential(
            nn.Conv2d(8 * patch_size, conv_hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.1),

            *[ResCnnBlock(conv_hidden_dim, rnn_dropout) for _ in range(cnn_layers)],
            nn.Flatten()
        )

        self.proj = nn.Linear(conv_hidden_dim * 64, self.n_units)

        self.gru = nn.GRU(
            input_size=self.n_units,
            hidden_size=self.n_units,
            num_layers = self.n_layers,
            dropout = self.rnn_dropout, 
            batch_first = True, # The first dim of our input is the batch dim
            bidirectional = False,
        )

        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)


        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x: torch.Tensor, day_idx, states=None, return_state = False):
        # input x shape: [B, T, F]
        B, T, _ = x.shape

        # Set up day specific transformation
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        # Einstien sum on day weights and x tensor
        # [T, F] x [512, 512] -> [B, T, 512]
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.training and self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # [B, T, 8, 8, 8] (8 of 8x8 brain electrode arrays)
        x = x.view(B, T, 8, 8, 8)

        # [B, T, 8, 8, 8, patch_size] "stack" feature maps on time axis using patch size and stride 
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        
        # [B, T, 8, patch_size, 8, 8]
        x = x.permute(0, 1, 2, 5, 3, 4).contiguous()

        num_windows = x.shape[1]

        # [B, T, 8 * patch_size, 8, 8]
        x = x.view(-1, 8 * self.patch_size, 8, 8)

        # [B * T, hidden_dim * 64]
        feats = self.tower(x)

        # [B, T, hidden_dim * 64]
        feats = feats.view(B, num_windows, -1)

        feats = self.proj(feats)

        if states is None:
            states = self.h0.expand(self.n_layers, B, self.n_units).contiguous()

        output, hidden_states = self.gru(feats, states)

        # [B, T, 41]
        logits = self.out(output)

        if return_state:
            return logits, hidden_states

        return logits


class CnnTransformerDecoder(nn.Module):
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 n_heads = 8,
                 dim_feedforward = 1024,
                 cnn_layers = 2,
                 transformer_dropout = 0.1,
                 conv_hidden_dim = 64,
                 input_dropout = 0.1,
                 n_layers = 4, 
                 patch_size = 10,
                 patch_stride = 4):
        
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days
        self.input_dropout = input_dropout
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride


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

        self.tower = nn.Sequential(
            nn.Conv2d(8 * patch_size, conv_hidden_dim, 3, 1, 1),
            nn.LeakyReLU(0.1),

            *[ResCnnBlock(conv_hidden_dim, transformer_dropout) for _ in range(cnn_layers)],
            nn.Flatten()
        )

        self.projection = nn.Linear(64 * conv_hidden_dim, n_units)

        self.pos_encoder = PositionalEncoder(n_units, dropout=transformer_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_units,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=False # PyTorch Transformer default is often [Seq, Batch, Dim]
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x: torch.Tensor, day_idx, src_key_padding_mask=None):
        # input x shape: [B, T, F]
        B, T, _ = x.shape

        # Set up day specific transformation
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        # Einstien sum on day weights and x tensor
        # [T, F] x [512, 512] -> [B, T, 512]
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.training and self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # [B, T, 8, 8, 8] (8 of 8x8 brain electrode arrays)
        x = x.view(B, T, 8, 8, 8)

        # [B, T, 8, 8, 8, patch_size] "stack" feature maps on time axis using patch size and stride 
        x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
        
        # [B, T, 8, patch_size, 8, 8]
        x = x.permute(0, 1, 2, 5, 3, 4).contiguous()

        num_windows = x.shape[1]

        # [B, T, 8 * patch_size, 8, 8]
        x = x.view(-1, 8 * self.patch_size, 8, 8)

        # [B * T, hidden_dim * 64]
        feats = self.tower(x)

        # [B, T, hidden_dim * 64]
        feats = feats.view(B, num_windows, -1)

        # [B, T, dim]
        feats = self.projection(feats)

        # [T, B, dim]
        feats = feats.permute(1, 0, 2)

        feats = self.pos_encoder(feats)

        encoded = self.transformer_encoder(feats, src_key_padding_mask=src_key_padding_mask)

        encoded = encoded.permute(1, 0, 2)

        logits = self.out(encoded)

        return logits


import os
    
if __name__ == "__main__":
    from rnn_trainer import BrainToTextDecoder_Trainer
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    config_path = os.path.join(script_dir, 'rnn_args.yaml')
    
    args = OmegaConf.load(config_path)

    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()
