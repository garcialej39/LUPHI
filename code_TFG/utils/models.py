"""
models.py — Definición de arquitecturas MIL para clasificación de remisión histológica
en colitis ulcerosa a partir de embeddings endoscópicos e histológicos.

Soporta:
Endoscopia:  BMCLIP (512), GastroNetViTs (384)
Histotología: CONCH  (512), KEEP (768)

-Modalidades soportadas: 'endo_only' | 'histo_only' | 'fusion'
-Fusión tardía: 'cat' | 'avg' | 'mhsa'
-Aggregators: 'ABMIL' | 'TransABMIL'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    def __init__(self, L, p=0.25):
        super().__init__()
        self.L = L
        self.D = L // 4
        self.K = 1
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Tanh(), nn.Dropout(p),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D), nn.Sigmoid(), nn.Dropout(p),
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, features):
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U)
        w = torch.softmax(A, dim=0)
        return torch.mm(features.T, w).squeeze()


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=128, dropout=True, n_classes=1):
        super().__init__()
        attention_a = [nn.Linear(L, D), nn.Tanh()]
        attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            attention_a.append(nn.Dropout(0.25))
            attention_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*attention_a)
        self.attention_b = nn.Sequential(*attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = self.attention_c(a * b)
        return A, x


class TransABMIL(nn.Module):
    def __init__(self, L=512, p=0.25, nhead=8):
        super().__init__()
        self.L = L
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=L, nhead=nhead, dim_feedforward=L,
            dropout=p, activation="relu", batch_first=False,
        )
        self.transformer    = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attention_head = Attn_Net_Gated(L=L, D=L // 4, dropout=True, n_classes=1)
        self.rho = nn.Sequential(nn.Linear(L, L), nn.ReLU(), nn.Dropout(p))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer(x)
        A, x    = self.attention_head(x)
        A = F.softmax(A.transpose(1, 0), dim=1)
        x = torch.matmul(A.transpose(2, 1), x.transpose(1, 0))
        return self.rho(x).squeeze()


class MILFusion(nn.Module):
    """
    MIL Fusion con projectors opcionales.

    Si endo_input_dim != L  → se añade un projector lineal para endo.
    Si histo_input_dim != L → se añade un projector lineal para histo.
    Esto permite usar cualquier combinación de los diferentes encoders
    (BMCLIP-512, GastroNetViTs-384, CONCH-512, KEEP-768)
    """
    def __init__(
        self,
        n_classes,
        L = 512,
        data_modality = "histo_only",
        late_fusion = "cat",
        aggregation = "TransABMIL",
        p = 0.25,
        use_l2_norm = True,
        endo_input_dim  = None,   
        histo_input_dim = None,   
    ):
        super().__init__()

        self.L  = L
        self.data_modality = data_modality
        self.late_fusion = late_fusion if data_modality == "fusion" else None
        self.use_l2_norm = use_l2_norm

        #Projectors (solo si la dim de entrada ≠ L)
        endo_dim  = endo_input_dim  if endo_input_dim  is not None else L
        histo_dim = histo_input_dim if histo_input_dim is not None else L

        self.proj_endo  = nn.Linear(endo_dim,  L) if endo_dim  != L else nn.Identity()
        self.proj_histo = nn.Linear(histo_dim, L) if histo_dim != L else nn.Identity()

        if endo_dim != L:
            print(f"[INFO] Endo projector:  {endo_dim} → {L}")
        if histo_dim != L:
            print(f"[INFO] Histo projector: {histo_dim} → {L}")

        #Aggregators
        Aggregator = TransABMIL if aggregation == "TransABMIL" else ABMIL

        self.MIL_endo  = Aggregator(L=L, p=p) if data_modality != "histo_only" else None
        self.MIL_histo = Aggregator(L=L, p=p) if data_modality != "endo_only"  else None

        #Fusion 
        if data_modality == "fusion" and late_fusion == "mhsa":
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=L, num_heads=8, dropout=p, batch_first=False,
            )
        else:
            self.multihead_attn = None

        classifier_input_dim = 2 * L if (data_modality == "fusion" and late_fusion == "cat") else L
        self.classifier = nn.Linear(classifier_input_dim, n_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features):
        if self.data_modality != "endo_only":
            features_histo = features[-1]
            if self.use_l2_norm:
                features_histo = F.normalize(features_histo, p=2, dim=-1)
            features_histo  = self.proj_histo(features_histo)
            embedding_histo = self.MIL_histo(features_histo)

        if self.data_modality != "histo_only":
            features_endo  = self.proj_endo(features[0])
            embedding_endo = self.MIL_endo(features_endo)

        if self.data_modality == "endo_only":
            embedding = embedding_endo
        elif self.data_modality == "histo_only":
            embedding = embedding_histo
        else:
            if self.late_fusion == "avg":
                embedding = (embedding_endo + embedding_histo) / 2
            elif self.late_fusion == "cat":
                embedding = torch.cat([embedding_endo, embedding_histo], dim=0)
            elif self.late_fusion == "mhsa":
                tokens = torch.stack([embedding_endo, embedding_histo], dim=0).unsqueeze(1)
                attended_tokens, _ = self.multihead_attn(tokens, tokens, tokens)
                embedding = attended_tokens.mean(dim=0).squeeze()
            else:
                raise ValueError(f"Unknown fusion method: {self.late_fusion}")

        return self.classifier(embedding)