import torch
import torch.nn as nn
import math
from vit_pytorch import ViT
import torch.nn as nn
import torch

class ImageSetTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(ImageSetTransformer, self).__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # Important to keep shape [batch, seq, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = input_dim

        # Optional: a learnable [CLS] token to summarize the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, input_dim]
        mask: [batch_size, seq_len] -> True for tokens to be masked
        """
        batch_size = x.size(0)

        # Add [CLS] token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1).to(x.device)  # Move to same device
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+seq_len, D]

        # Extend mask to account for CLS (CLS not masked)
        if mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat((cls_mask, mask), dim=1)

        # Transformer pass
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        return x[:, 0, :]  # Return CLS token embedding


class CaloricRegressor(nn.Module):
    def __init__(self, cgm_emb_size, activity_emb_size, meal_timing_emb_size, 
                 demographics_size=5, hidden_size=128, output_size=1):
        super(CaloricRegressor, self).__init__()
        
        total_input_size = cgm_emb_size + activity_emb_size + meal_timing_emb_size + demographics_size
        
        self.regressor = nn.Sequential(
            nn.Linear(total_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        return self.regressor(x)
    
class MealTimingEncoder(nn.Module):
    def __init__(self, input_channels=5, hidden_size=64, output_size=32):
        super(MealTimingEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_size, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size, hidden_size*2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size*2, hidden_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        self.fc = nn.Linear(hidden_size*4, output_size)
        
    def forward(self, x):
        # x shape: [batch_size, channels=5, seq_len=1440]
        x = self.encoder(x)
        x = x.squeeze(-1)  # Remove the last dimension after global pooling
        x = self.fc(x)
        return x
    

class CaloricModel(nn.Module):
    def __init__(self):
        super(CaloricModel, self).__init__()
        self.cgm_encoder = MultiheadAttention(n_features=1440, embed_dim=96, num_heads=2, num_classes=64, dropout=0.2, num_layers=6)
        self.activity_encoder = MultiheadAttention(n_features=1440, embed_dim=96, num_heads=2, num_classes=64, dropout=0.2, num_layers=6)
        self.meal_timing_encoder = MealTimingEncoder(input_channels=5, hidden_size=64, output_size=32)
        self.caloric_regressor = CaloricRegressor(
            cgm_emb_size=64,
            activity_emb_size=64,
            meal_timing_emb_size=32,
            demographics_size=5
        )

    def forward(self, cgm, activity, meal_timing, demographics):
        cgm_embed = self.cgm_encoder(cgm)
        activity_embed = self.activity_encoder(activity)
        meal_embed = self.meal_timing_encoder(meal_timing)

        # Concatenate all embeddings and demographic vector
        x = torch.cat([cgm_embed, activity_embed, meal_embed, demographics], dim=1)
        return self.caloric_regressor(x)
    



class ImageEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=32, channels=3, dropout=0.2):
        super(ImageEncoder, self).__init__()
        self.output_dim = num_classes
        
        # Vision Transformer for feature extraction
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=dropout,
            emb_dropout=dropout,
            channels=channels
        )
        
    def forward(self, x):
        return self.vit(x)

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        n_features,
        embed_dim,
        num_heads,
        num_classes,
        dropout=0,
        num_layers=6,
        use_pos_emb=False,
        activation=nn.GELU(),
    ):
        super().__init__()

        self.use_pos_emb = use_pos_emb

        self.conv = nn.Conv1d(n_features, embed_dim, 1, 1)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embed_dim,
                num_heads,
                batch_first=True,
                dropout=dropout,
                activation=activation,
            ),
            num_layers,
        )

        self.register_buffer(
            "position_vec",
            torch.tensor(
                [
                    math.pow(10000.0, 2.0 * (i // 2) / embed_dim)
                    for i in range(embed_dim)
                ],
            ),
        )

        self.linear = nn.Linear(embed_dim, num_classes)
        self.sig = nn.Sigmoid()

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time[:, None] / self.position_vec[None, :, None]
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask[:, None]

    def forward(self, x, lens=0, t=0):
        mask = (x == float("inf"))[:, :, 0]
        x[mask] = 0

        z = self.conv(x.permute(0, 2, 1))

        if self.use_pos_emb:
            tem_enc = self.temporal_enc(t, mask)
            z = z + tem_enc

        z = z.permute(0, 2, 1).float()

        z = self.attn(z, src_key_padding_mask=mask)

        return self.linear(z.mean(1))