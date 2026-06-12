"""
main.py — MIL Framework PICASSO
Soporta encoders de distintas dimensiones mediante projectors:
    Endo:  BMCLIP (512), GastroNetViTs (384)
    Histo: CONCH  (512), KEEP (768)
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.models import MILFusion
from utils.my_utils import (
    set_random_seeds, plot_confmx, load_data,
    aggregate_patient_predictions, compute_binary_metrics, find_optimal_threshold,
)
from utils.trainer import train_model, validate_model

# Argumentos
parser = argparse.ArgumentParser(description="MIL framework — PICASSO dataset")
parser.add_argument("--folder",         type=str,   default="/workspace/test_docker/data/embeddings/")
parser.add_argument("--dataset_file",   type=str,   default="/workspace/test_docker/data/PICASSO_dataframe.xlsx")
parser.add_argument("--target",         type=str,   default="R_HIST", choices=["R_HIST", "R_ENDOS"])
parser.add_argument("--data_modality",  type=str,   default="histo_only",
                    choices=["histo_only", "endo_only", "fusion"])
parser.add_argument("--late_fusion",    type=str,   default="cat", choices=["avg", "cat", "mhsa"])
parser.add_argument("--aggregator",     type=str,   default="TransABMIL", choices=["ABMIL", "TransABMIL"])
parser.add_argument("--histo_encoder",  type=str,   default="CONCH",
                    help="CONCH (512) | KEEP (768)")
parser.add_argument("--endo_encoder",   type=str,   default="BMCLIP",
                    help="BMCLIP (512) | GastroNetViTs (384)")
parser.add_argument("--L",              type=int,   default=512,
                    help="Dimensión interna del modelo (projectors → L)")
parser.add_argument("--k_folds",        type=int,   default=5)
parser.add_argument("--lr",             type=float, default=1e-4)
parser.add_argument("--epochs",         type=int,   default=50)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--no_l2_norm",     action="store_true")

args = parser.parse_args()
use_l2_norm = not args.no_l2_norm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

set_random_seeds(args.seed)

# Run name
l2_tag = "l2norm" if use_l2_norm else "nol2norm"

if args.data_modality == "histo_only":
    run_name = f"histo_only_{args.histo_encoder}_{args.aggregator}_{args.target}_{l2_tag}_seed{args.seed}"
elif args.data_modality == "endo_only":
    run_name = f"endo_only_{args.endo_encoder}_{args.aggregator}_{args.target}_{l2_tag}_seed{args.seed}"
else:
    run_name = (f"fusion_{args.endo_encoder}_{args.histo_encoder}_"
                f"{args.aggregator}_{args.late_fusion}_{args.target}_{l2_tag}_seed{args.seed}")

print(f"\n{'='*70}")
print(f"Run       : {run_name}")
print(f"Device    : {device}")
print(f"L2 histo  : {'ON' if use_l2_norm else 'OFF'}")
print(f"{'='*70}\n")


# Datos
data = load_data(
    dataset_file  = args.dataset_file,
    target = args.target,
    endo_encoder = args.endo_encoder,
    histo_encoder = args.histo_encoder,
    folder = args.folder,
    data_modality = args.data_modality,
)

Y  = np.array([d[2] for d in data])
groups = np.array([d[3] for d in data])
n_classes = len(np.unique(Y))

# Detectar dimensiones reales de los encoders
if args.data_modality == "fusion":
    endo_input_dim  = data[0][0].shape[1]
    histo_input_dim = data[0][1].shape[1]
elif args.data_modality == "histo_only":
    endo_input_dim  = None
    histo_input_dim = data[0][1].shape[1]
else:
    endo_input_dim  = data[0][0].shape[1]
    histo_input_dim = None

print(f"[INFO] Total samples : {len(data)}")
print(f"[INFO] Unique patients   : {len(np.unique(groups))}")
print(f"[INFO] Class distribution: {np.bincount(Y)}")
print(f"[INFO] Endo input dim : {endo_input_dim}")
print(f"[INFO] Histo input dim : {histo_input_dim}")
print(f"[INFO] Internal L : {args.L}\n")

# Cross-validation
kf = StratifiedGroupKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

all_sample_predictions = []
all_patient_predictions = []
fold_metrics = []

for fold, (train_index, val_index) in enumerate(kf.split(np.zeros(len(Y)), Y, groups)):

    print(f"\n{'#'*25} Fold {fold+1}/{args.k_folds} {'#'*25}")
    run_name_k = f"{run_name}_fold{fold}"

    X_train = [data[i] for i in train_index]
    X_val = [data[i] for i in val_index]
    Y_train = Y[train_index]
    Y_val = Y[val_index]

    print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"[INFO] Train class dist: {np.bincount(Y_train)}")

    model = MILFusion(
        n_classes = n_classes,
        L = args.L,
        late_fusion = args.late_fusion,
        data_modality = args.data_modality,
        aggregation = args.aggregator,
        use_l2_norm = use_l2_norm,
        endo_input_dim = endo_input_dim,
        histo_input_dim = histo_input_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(Y_train), y=Y_train,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device), reduction="mean",
    )

    train_model(
        model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
        train_data=X_train, train_labels=Y_train, val_data=X_val, val_labels=Y_val,
        epochs=args.epochs, run_name=run_name_k, data_modality=args.data_modality, device=device,
    )

    sample_preds = validate_model(
        model=model, val_data=X_val, val_labels=Y_val,
        data_modality=args.data_modality, device=device,
    )

    # Guardar pesos teacher para KD
    if args.data_modality == "histo_only":
        os.makedirs("./local_data/results", exist_ok=True)
        torch.save(model.MIL_histo.state_dict(),
                   f"./local_data/results/teacher_histo_{run_name_k}.pt")
        torch.save(model.classifier.state_dict(),
                   f"./local_data/results/teacher_clf_{run_name_k}.pt")
        # Guardar también el projector si existe
        if not isinstance(model.proj_histo, torch.nn.Identity):
            torch.save(model.proj_histo.state_dict(),
                       f"./local_data/results/teacher_proj_histo_{run_name_k}.pt")
            print(f"[INFO] Teacher proj_histo saved: fold {fold}")
        print(f"[INFO] Teacher weights saved: fold {fold}")

    for p in sample_preds:
        p.update({"fold": fold, "run_name": run_name, "modality": args.data_modality,
                  "target": args.target, "seed": args.seed,
                  "late_fusion": args.late_fusion if args.data_modality == "fusion" else "none",
                  "aggregator": args.aggregator, "l2_norm": use_l2_norm})

    all_sample_predictions.extend(sample_preds)

    patient_preds_fold = aggregate_patient_predictions(sample_preds)
    opt_threshold, opt_bal_acc = find_optimal_threshold(
        y_true=patient_preds_fold["true_label"].values,
        y_prob=patient_preds_fold["prob_class1"].values,
    )
    print(f"[FOLD {fold}] Optimal threshold={opt_threshold:.2f} (BAL_ACC={opt_bal_acc:.4f})")

    patient_preds_fold["pred_label"] = (patient_preds_fold["prob_class1"] >= opt_threshold).astype(int)
    patient_preds_fold["threshold"] = opt_threshold
    patient_preds_fold["fold"] = fold
    patient_preds_fold["run_name"] = run_name
    all_patient_predictions.append(patient_preds_fold)

    fold_metric = compute_binary_metrics(
        y_true=patient_preds_fold["true_label"].values,
        y_prob=patient_preds_fold["prob_class1"].values,
        threshold=opt_threshold,
    )
    fold_metric.update({"fold": fold, "n_patients": len(patient_preds_fold)})
    fold_metrics.append(fold_metric)
    plot_confmx(fold_metric["confusion_matrix"], run_name_k)

    print(f"[FOLD {fold}] BAL_ACC={fold_metric['BAL_ACC']:.4f} | "
          f"AUC={fold_metric['AUC']:.4f} | "
          f"Sens={fold_metric['Sensitivity']:.4f} | "
          f"Spec={fold_metric['Specificity']:.4f}")

    torch.cuda.empty_cache()

# Evaluación final
sample_preds_df  = pd.DataFrame(all_sample_predictions)
patient_preds_df = pd.concat(all_patient_predictions, ignore_index=True)

opt_threshold_global, _ = find_optimal_threshold(
    y_true=patient_preds_df["true_label"].values,
    y_prob=patient_preds_df["prob_class1"].values,
)

metrics_dict = compute_binary_metrics(
    y_true=patient_preds_df["true_label"].values,
    y_prob=patient_preds_df["prob_class1"].values,
    threshold=opt_threshold_global,
)
metrics_dict.update({
    "run_name": run_name, "modality": args.data_modality, "target": args.target,
    "late_fusion": args.late_fusion if args.data_modality == "fusion" else "none",
    "aggregator": args.aggregator, "l2_norm": use_l2_norm, "seed": args.seed,
    "endo_encoder": args.endo_encoder, "histo_encoder": args.histo_encoder,
    "n_samples": len(sample_preds_df), "n_patients": len(patient_preds_df),
    "mean_fold_threshold": round(float(np.mean([fm["threshold"] for fm in fold_metrics])), 2),
})

cfmx = metrics_dict.pop("confusion_matrix")
plot_confmx(cfmx, run_name)

print(f"\n{'='*70}")
print(f"FINAL METRICS — {run_name}")
for k, v in metrics_dict.items():
    print(f"  {k:<25}: {v}")
print(f"  Confusion matrix:\n{cfmx}")

fold_auc = [fm["AUC"] for fm in fold_metrics]
fold_bac = [fm["BAL_ACC"] for fm in fold_metrics]
print("\nPer-fold:")
for fm in fold_metrics:
    print(f"  Fold {fm['fold']}: BAL_ACC={fm['BAL_ACC']:.4f} | AUC={fm['AUC']:.4f}")
print(f"  Mean BAL_ACC: {np.mean(fold_bac):.4f} ± {np.std(fold_bac):.4f}")
print(f"  Mean AUC    : {np.mean(fold_auc):.4f} ± {np.std(fold_auc):.4f}")
print(f"{'='*70}\n")

# Guardar
os.makedirs("./local_data/results", exist_ok=True)
pd.DataFrame([metrics_dict]).to_excel(f"./local_data/results/metrics_{run_name}.xlsx", index=False)
pd.DataFrame(fold_metrics).drop(columns=["confusion_matrix"], errors="ignore").to_excel(
    f"./local_data/results/fold_metrics_{run_name}.xlsx", index=False)
pd.DataFrame(all_sample_predictions).to_excel(
    f"./local_data/results/sample_predictions_{run_name}.xlsx", index=False)

for col, val in [("modality", args.data_modality), ("target", args.target),
                 ("late_fusion", args.late_fusion if args.data_modality == "fusion" else "none"),
                 ("aggregator", args.aggregator), ("l2_norm", use_l2_norm), ("seed", args.seed),
                 ("endo_encoder", args.endo_encoder), ("histo_encoder", args.histo_encoder)]:
    patient_preds_df[col] = val

patient_preds_df.to_excel(f"./local_data/results/patient_predictions_{run_name}.xlsx", index=False)

metrics_csv = "./local_data/results/all_metrics.csv"
metrics_df  = pd.DataFrame([metrics_dict])
if os.path.exists(metrics_csv):
    existing   = pd.read_csv(metrics_csv)
    existing   = existing[existing["run_name"] != run_name]
    metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
metrics_df.to_csv(metrics_csv, index=False)

print("[DONE]")