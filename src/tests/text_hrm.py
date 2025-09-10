import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, nhead=4, dim_feedforward=256, dropout=0.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward,
                                               batch_first=True, dropout=dropout,
                                               activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        return self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)

class HRMText(nn.Module):
    """
    HRM-style minimalista para texto:
      - Token embedding + token especial [CLS]
      - Módulo L: T passos por ciclo (rápido)
      - Módulo H: 1 passo por ciclo (lento)
      - N ciclos
      - Pooling no [CLS] do H -> 3 cabeças (autor, tipo, ano)
    """
    def __init__(self, vocab_size, d_model=128, steps=3, cycles=3,
                 n_authors=3, n_types=3, n_years=6, pad_id=0):
        super().__init__()
        self.steps = steps
        self.cycles = cycles
        self.pad_id = pad_id
        self.cls_id = vocab_size  # id virtual do [CLS]

        self.embed = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_id)  # +1 para [CLS]
        self.L = TransformerBlock(d_model=d_model, nhead=4, dim_feedforward=4*d_model, dropout=0.0)
        self.H = TransformerBlock(d_model=d_model, nhead=4, dim_feedforward=4*d_model, dropout=0.0)

        self.head_author = nn.Linear(d_model, n_authors)
        self.head_type   = nn.Linear(d_model, n_types)
        self.head_year   = nn.Linear(d_model, n_years)

    def forward(self, x_ids, lengths):
        """
        x_ids: (B, L) ids dos tokens (sem [CLS]); inserimos [CLS] no início
        lengths: (B,) tamanhos reais (sem pad)
        """
        B, L = x_ids.shape
        device = x_ids.device

        # Prepend [CLS]
        cls_col = torch.full((B, 1), self.cls_id, dtype=torch.long, device=device)
        x_ids_cls = torch.cat([cls_col, x_ids], dim=1)  # (B, L+1)

        # Masks
        pad_mask = (x_ids_cls == self.pad_id)  # (B, L+1)
        x = self.embed(x_ids_cls)              # (B, L+1, D)

        # Estados iniciais
        zL = torch.zeros_like(x)
        zH = torch.zeros_like(x)

        for _ in range(self.cycles):
            for _t in range(self.steps):
                zL = self.L(zL + x + zH, key_padding_mask=pad_mask)
            zH = self.H(zH + zL, key_padding_mask=pad_mask)

        cls_vec = zH[:, 0, :]  # (B, D)

        logits_author = self.head_author(cls_vec)
        logits_type   = self.head_type(cls_vec)
        logits_year   = self.head_year(cls_vec)

        return logits_author, logits_type, logits_year
