import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from functools import partial

# Import KAT_Group from kat_rational
from kat_rational import KAT_Group

# Custom MLP layer using KAT_Group activation
class KAN(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=KAT_Group,
            norm_layer=None,
            bias=True,
            drop=0.,
            act_init="gelu",
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias
        drop_probs = drop if isinstance(drop, tuple) else (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act1 = KAT_Group(mode="identity")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.act2 = KAT_Group(mode=act_init)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x

# Transformer Encoder Block
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: nn.Module = KAT_Group,
            norm_layer: nn.Module = nn.LayerNorm,
            act_init: str = 'gelu',
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.drop_path1 = nn.Identity()  # DropPath not essential for now

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = KAN(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
            act_init=act_init,
        )
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        # Self-attention
        x_attn = self.attn(x, x, x)[0]
        x = x + self.drop_path1(x_attn)
        x = self.norm1(x)

        # MLP
        x_mlp = self.mlp(x)
        x = x + self.drop_path2(x_mlp)
        x = self.norm2(x)
        return x

# Custom KATTransformer for sequence data
class KATTransformer(nn.Module):
    def __init__(
            self,
            input_dim=8*65,
            d_model=256,
            num_heads=8,
            num_layers=6,
            num_frames=500,
            output_dim=3,
            act_init='gelu',
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_frames = num_frames

        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(num_frames, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=d_model,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                act_layer=KAT_Group,
                norm_layer=nn.LayerNorm,
                act_init=act_init,
            )
            for _ in range(num_layers)
        ])

        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        batch_size, num_channels, num_frames, num_features = x.shape

        # Reshape input to [batch_size, num_frames, input_dim]
        x = x.view(batch_size, num_frames, -1)

        # Input embedding
        x = self.input_embedding(x)  # [batch_size, num_frames, d_model]

        # Add positional encoding
        x = x + self.positional_encoding[:num_frames, :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Output layer
        output = self.fc_out(x)  # [batch_size, num_frames, output_dim]
        return output

# Example usage
if __name__ == "__main__":
    # Dummy input
    dummy_input = torch.randn(128, 8, 500, 65)
    
    # Initialize model
    model = KATTransformer(
        input_dim=8*65,
        d_model=256,
        num_heads=8,
        num_layers=6,
        num_frames=500,
        output_dim=3,
        act_init='gelu',
    )

    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()

    output = model(dummy_input)
    print(output)
    print(output.shape)
