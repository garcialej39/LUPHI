"""
models_kd.py — KD Hinton con soporte para diferentes encoders (projectors)

Soporta:
Endoscopia:  BMCLIP (512), GastroNetViTs (384)
Histotología: CONCH  (512), KEEP (768)

-Aggregators soportados: 'ABMIL' | 'TransABMIL'
-Hiperparámetros KD: temperatura T, α 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.models import ABMIL, TransABMIL


class KDMILModel(nn.Module):
    def __init__(
        self,
        n_classes,
        L  = 512,
        aggregation = "TransABMIL",
        freeze_teacher  = True,
        p = 0.25,
        use_l2_norm_histo = False,
        temperature = 4.0,
        alpha  = 0.5,
        endo_input_dim = None,  
        histo_input_dim = None,   
    ):
        super().__init__()

        self.L = L
        self.n_classes = n_classes
        self.freeze_teacher = freeze_teacher
        self.use_l2_norm_histo = use_l2_norm_histo
        self.temperature = temperature
        self.alpha = alpha

        endo_dim = endo_input_dim  if endo_input_dim  is not None else L
        histo_dim = histo_input_dim if histo_input_dim is not None else L

        Agg = TransABMIL if aggregation == "TransABMIL" else ABMIL

        #Projectors
        self.proj_endo  = nn.Linear(endo_dim,  L) if endo_dim  != L else nn.Identity()
        self.proj_histo = nn.Linear(histo_dim, L) if histo_dim != L else nn.Identity()

        if endo_dim != L:
            print(f"[INFO] Student endo projector:  {endo_dim} → {L}")
        if histo_dim != L:
            print(f"[INFO] Teacher histo projector: {histo_dim} → {L}")

        #Student (endoscopia)
        self.student_agg = Agg(L=L, p=p)
        self.student_clf = nn.Linear(L, n_classes)
        nn.init.xavier_uniform_(self.student_clf.weight)
        nn.init.zeros_(self.student_clf.bias)

        #Teacher (histología)
        self.teacher_agg = Agg(L=L, p=p)
        self.teacher_clf = nn.Linear(L, n_classes)
        nn.init.xavier_uniform_(self.teacher_clf.weight)
        nn.init.zeros_(self.teacher_clf.bias)

        if freeze_teacher:
            for p_ in self.teacher_agg.parameters():
                p_.requires_grad = False
            for p_ in self.teacher_clf.parameters():
                p_.requires_grad = False

    def load_teacher_weights(self, agg_state_dict, clf_state_dict=None):
        self.teacher_agg.load_state_dict(agg_state_dict)
        if clf_state_dict is not None:
            self.teacher_clf.load_state_dict(clf_state_dict)
        if self.freeze_teacher:
            for p_ in self.teacher_agg.parameters():
                p_.requires_grad = False
            for p_ in self.teacher_clf.parameters():
                p_.requires_grad = False

    def forward(self, features_endo, features_histo=None):
        # projector endo a L
        features_endo = self.proj_endo(features_endo)
        emb_student = self.student_agg(features_endo)
        student_logits = self.student_clf(emb_student)

        teacher_logits = None
        if features_histo is not None:
            if self.use_l2_norm_histo:
                features_histo = F.normalize(features_histo, p=2, dim=-1)
            # projector histo a L
            features_histo = self.proj_histo(features_histo)

            if self.freeze_teacher:
                with torch.no_grad():
                    emb_teacher = self.teacher_agg(features_histo)
                    teacher_logits = self.teacher_clf(emb_teacher)
            else:
                emb_teacher = self.teacher_agg(features_histo)
                teacher_logits = self.teacher_clf(emb_teacher)

        return student_logits, teacher_logits

    def hinton_kd_loss(self, student_logits, teacher_logits):
        T = self.temperature
        p_s = F.log_softmax(student_logits.unsqueeze(0) / T, dim=1)
        p_t = F.softmax(teacher_logits.unsqueeze(0) / T, dim=1)
        return F.kl_div(p_s, p_t, reduction="batchmean") * (T ** 2)

    def total_loss(self, student_logits, teacher_logits, label, criterion):
        loss_ce = criterion(student_logits.unsqueeze(0), label.unsqueeze(0))
        loss_kd = self.hinton_kd_loss(student_logits, teacher_logits)
        loss    = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd
        return loss, loss_ce, loss_kd