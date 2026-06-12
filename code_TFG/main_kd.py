"""
main_kd.py — KD Hinton con soporte de diferentes encoders.
Loss = α·CE + (1-α)·T²·KL(student/T ‖ teacher/T)

Soporta:
Endoscopia:  BMCLIP (512), GastroNetViTs (384)
Histología: CONCH  (512), KEEP (768)
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.models_kd import KDMILModel
from utils.my_utils import (
    set_random_seeds, plot_confmx, load_data,
    aggregate_patient_predictions, compute_binary_metrics, find_optimal_threshold,
)
from utils.trainer_kd import train_model_kd, validate_model_kd

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--folder",         type=str,   default="/workspace/test_docker/data/embeddings/")
parser.add_argument("--dataset_file",   type=str,   default="/workspace/test_docker/data/PICASSO_dataframe.xlsx")
parser.add_argument("--target",         type=str,   default="R_HIST", choices=["R_HIST", "R_ENDOS"])
parser.add_argument("--aggregator",     type=str,   default="TransABMIL", choices=["ABMIL", "TransABMIL"])
parser.add_argument("--histo_encoder",  type=str,   default="CONCH",
                    help="CONCH (512) | KEEP (768)")
parser.add_argument("--endo_encoder",   type=str,   default="BMCLIP",
                    help="BMCLIP (512) | GastroNetViTs (384)")
parser.add_argument("--L",              type=int,   default=512,
                    help="Dimensión interna del modelo")
parser.add_argument("--k_folds",        type=int,   default=5)
parser.add_argument("--lr",             type=float, default=1e-4)
parser.add_argument("--epochs",         type=int,   default=50)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--no_l2_norm",     action="store_true")
parser.add_argument("--temperature",    type=float, default=4.0)
parser.add_argument("--alpha",          type=float, default=0.25)
parser.add_argument("--freeze_teacher", action="store_true", default=True)
parser.add_argument("--no_freeze_teacher", action="store_false", dest="freeze_teacher")
parser.add_argument("--pretrained_dir", type=str,   default="./local_data/results")

args = parser.parse_args()
use_l2_norm_histo = not args.no_l2_norm

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

set_random_seeds(args.seed)


# Run name
l2_tag = "l2" if use_l2_norm_histo else "nol2"
freeze_tag = "frozen" if args.freeze_teacher else "joint"

run_name = (
    f"hinton_kd_{args.endo_encoder}_{args.histo_encoder}_"
    f"{args.aggregator}_{args.target}_{l2_tag}_"
    f"T{args.temperature}_a{args.alpha}_{freeze_tag}_seed{args.seed}"
)

print(f"\n{'='*70}")
print(f"Run : {run_name}")
print(f"Device : {device}")
print(f"Endo encoder  : {args.endo_encoder}")
print(f"Histo encoder : {args.histo_encoder}")
print(f"Temperature T : {args.temperature}")
print(f"Alpha (CE)    : {args.alpha} | Alpha (KD): {1-args.alpha:.2f}")
print(f"{'='*70}\n")

# Datos
data = load_data(
    dataset_file  = args.dataset_file,
    target = args.target,
    endo_encoder  = args.endo_encoder,
    histo_encoder = args.histo_encoder,
    folder = args.folder,
    data_modality = "fusion",
)

Y  = np.array([d[2] for d in data])
groups = np.array([d[3] for d in data])
n_classes = len(np.unique(Y))

# Detectar dimensiones reales automáticamente
endo_input_dim = data[0][0].shape[1]
histo_input_dim = data[0][1].shape[1]

print(f"[INFO] Total samples : {len(data)}")
print(f"[INFO] Unique patients : {len(np.unique(groups))}")
print(f"[INFO] Class dist : {np.bincount(Y)}")
print(f"[INFO] Endo input dim  : {endo_input_dim}")
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
    X_val   = [data[i] for i in val_index]
    Y_train = Y[train_index]

    print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"[INFO] Train class dist: {np.bincount(Y_train)}")

    # Modelo
    model = KDMILModel(
        n_classes = n_classes,
        L = args.L,
        aggregation = args.aggregator,
        freeze_teacher = args.freeze_teacher,
        use_l2_norm_histo = use_l2_norm_histo,
        temperature = args.temperature,
        alpha  = args.alpha,
        endo_input_dim = endo_input_dim,
        histo_input_dim = histo_input_dim,
    ).to(device)

    #Pesos teacher 
    l2_tag_pt = "l2norm" if use_l2_norm_histo else "nol2norm"
    teacher_agg_path = (
        f"{args.pretrained_dir}/"
        f"teacher_histo_histo_only_{args.histo_encoder}_"
        f"{args.aggregator}_{args.target}_{l2_tag_pt}_"
        f"seed{args.seed}_fold{fold}.pt"
    )
    teacher_clf_path = teacher_agg_path.replace("teacher_histo_", "teacher_clf_")
    teacher_proj_path = teacher_agg_path.replace("teacher_histo_", "teacher_proj_histo_")

    if os.path.exists(teacher_agg_path):
        agg_state = torch.load(teacher_agg_path, map_location=device)
        clf_state = torch.load(teacher_clf_path, map_location=device) \
                    if os.path.exists(teacher_clf_path) else None
        model.load_teacher_weights(agg_state, clf_state)

        #Cargar projector del teacher si existe (por ej. KEEP 768→512)
        if os.path.exists(teacher_proj_path) and not isinstance(model.proj_histo, nn.Identity):
            proj_state = torch.load(teacher_proj_path, map_location=device)
            model.proj_histo.load_state_dict(proj_state)
            for p_ in model.proj_histo.parameters():
                p_.requires_grad = False
            print(f"[INFO] Teacher proj_histo loaded and frozen: fold {fold}")

        print(f"[INFO] Teacher loaded: fold {fold}")
    else:
        print(f"[WARN] Teacher not found: {teacher_agg_path}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params:,}")

    #Optimizer 
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-5,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(Y_train), y=Y_train,
    )
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        reduction="mean",
    )

    #Entrenamiento
    train_model_kd(
        model=model, optimizer=optimizer, criterion=criterion,
        scheduler=scheduler, train_data=X_train, val_data=X_val,
        epochs=args.epochs, run_name=run_name_k, device=device,
    )

    #Predicciones
    sample_preds = validate_model_kd(model=model, val_data=X_val, device=device)

    for p in sample_preds:
        p.update({"fold": fold, "run_name": run_name, "modality": "hinton_kd_endo_only",
                  "target": args.target, "seed": args.seed, "aggregator": args.aggregator,
                  "temperature": args.temperature, "alpha": args.alpha,
                  "endo_encoder": args.endo_encoder, "histo_encoder": args.histo_encoder})

    all_sample_predictions.extend(sample_preds)

    patient_preds_fold = aggregate_patient_predictions(sample_preds)
    opt_threshold, opt_bal_acc = find_optimal_threshold(
        y_true=patient_preds_fold["true_label"].values,
        y_prob=patient_preds_fold["prob_class1"].values,
    )
    print(f"[FOLD {fold}] Optimal threshold={opt_threshold:.2f} (BAL_ACC={opt_bal_acc:.4f})")

    patient_preds_fold["pred_label"] = (patient_preds_fold["prob_class1"] >= opt_threshold).astype(int)
    patient_preds_fold["threshold"]  = opt_threshold
    patient_preds_fold["fold"] = fold
    patient_preds_fold["run_name"] = run_name
    all_patient_predictions.append(patient_preds_fold)

    fold_metric = compute_binary_metrics(
        y_true=patient_preds_fold["true_label"].values,
        y_prob=patient_preds_fold["prob_class1"].values,
        threshold=opt_threshold,
    )
    fold_metric["fold"] = fold
    fold_metrics.append(fold_metric)
    plot_confmx(fold_metric["confusion_matrix"], run_name_k)

    print(f"[FOLD {fold}] BAL_ACC={fold_metric['BAL_ACC']:.4f} | "
          f"AUC={fold_metric['AUC']:.4f} | "
          f"Sens={fold_metric['Sensitivity']:.4f} | "
          f"Spec={fold_metric['Specificity']:.4f}")

    torch.cuda.empty_cache()

# Evaluación final
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
    "run_name": run_name, "modality": "hinton_kd_endo_only",
    "target": args.target, "aggregator": args.aggregator,
    "temperature": args.temperature, "alpha": args.alpha,
    "endo_encoder": args.endo_encoder, "histo_encoder": args.histo_encoder,
    "seed": args.seed, "n_patients": len(patient_preds_df),
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
patient_preds_df.to_excel(
    f"./local_data/results/patient_predictions_{run_name}.xlsx", index=False)

metrics_csv = "./local_data/results/all_metrics.csv"
metrics_df  = pd.DataFrame([metrics_dict])
if os.path.exists(metrics_csv):
    existing   = pd.read_csv(metrics_csv)
    existing   = existing[existing["run_name"] != run_name]
    metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
metrics_df.to_csv(metrics_csv, index=False)

print("[DONE]")