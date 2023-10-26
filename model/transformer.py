import torch.nn as nn
from model.patch_embed import PatchEmbedding
from model.transformer_layer import TransformerLayer


class VIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_layers = config['n_layers']
        emb_dim = config['emb_dim']
        num_classes = config['num_classes']
        self.patch_embed_layer = PatchEmbedding(config)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.fc_number = nn.Linear(emb_dim, num_classes)
        
    def forward(self, x):
        # Patchify and add CLS token
        out = self.patch_embed_layer(x)
        
        # Go through the transformer layers
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        
        # Compute logits
        return self.fc_number(out[:, 0])
