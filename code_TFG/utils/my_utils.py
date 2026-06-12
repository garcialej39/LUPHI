import os
import glob
import re
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
)


def _build_endo_map(endo_folder):
    mapping = {}

    pattern = re.compile(
        r"^(\d{2})-(\d{2})\s+(rectum|sigmoid)",
        re.IGNORECASE | re.UNICODE,
    )

    for fpath in glob.glob(os.path.join(endo_folder, "*.npy")):
        fname = os.path.basename(fpath)
        match = pattern.match(fname)

        if match is None:
            continue

        center = match.group(1)
        patient = match.group(2)
        region = match.group(3).lower()

        section = "1" if region == "rectum" else "2"
        key = f"pat{center}{patient}_section{section}"

        if key in mapping:
            print(f"[WARN] Duplicate endoscopy key {key}. Keeping first file.")
            continue

        mapping[key] = fpath

    return mapping


def _load_patient_labels(dataset_file):
    raw = pd.read_excel(dataset_file, sheet_name=1, header=None)

    data_rows = raw.iloc[3:].copy()
    data_rows.columns = range(raw.shape[1])

    data_rows = data_rows[data_rows[0].notna()]
    data_rows = data_rows[
        data_rows[0].astype(str).str.strip() != "Total general"
    ]

    label_map = {}

    for _, row in data_rows.iterrows():
        pat_id = str(row[0]).strip()
        r_endos = row[3]
        r_hist = row[4]

        if pd.isna(r_endos) or pd.isna(r_hist):
            print(f"[WARN] Missing label for patient {pat_id}. Skipping.")
            continue

        label_map[pat_id] = {
            "R_ENDOS": int(r_endos),
            "R_HIST" : int(r_hist),
        }

    return label_map


def load_data(
    dataset_file,
    target,
    endo_encoder,
    histo_encoder,
    folder,
    data_modality="histo_only",
):
    if target not in ["R_HIST", "R_ENDOS"]:
        raise ValueError(f"Unknown target: {target}")

    label_map = _load_patient_labels(dataset_file)
    print(f"[INFO] Loaded labels for {len(label_map)} patients.")

    dataframe = pd.read_excel(dataset_file, sheet_name=0)
    list_video = dataframe["Video"].values
    list_wsi = dataframe["WSI"].values
    list_pat = dataframe["PatID"].values

    need_endo  = data_modality in ["endo_only", "fusion"]
    need_histo = data_modality in ["histo_only", "fusion"]

    endo_map = {}
    endo_map_by_name = {}

    if need_endo:
        endo_folder = os.path.join(folder, "endoscopy", endo_encoder)
        endo_map = _build_endo_map(endo_folder)

        for fpath in glob.glob(os.path.join(endo_folder, "*.npy")):
            key = os.path.splitext(os.path.basename(fpath))[0].lower().strip()
            endo_map_by_name[key] = fpath

        print(f"[INFO] Endoscopy embeddings found: {len(endo_map_by_name)}")

    data  = []
    skipped = 0

    for i, video_name in enumerate(tqdm(list_video, desc="Loading embeddings")):
        pat_id = str(list_pat[i]).strip()

        if pat_id not in label_map:
            print(f"[WARN] Patient {pat_id} not found in label sheet.")
            skipped += 1
            continue

        label = label_map[pat_id][target]

        if need_endo:
            video_key = str(video_name).strip()

            if re.match(r"^pat\d{4}_section\d$", video_key):
                fn_endo = endo_map.get(video_key)
            else:
                fn_endo = endo_map_by_name.get(video_key.lower())

            if fn_endo is None:
                print(f"[WARN] Missing endoscopy file for {video_key}")
                skipped += 1
                continue
        else:
            fn_endo = None

        if need_histo:
            fn_histo = os.path.join(
                folder, "histology", histo_encoder,
                str(list_wsi[i]) + ".npy",
            )
            if not os.path.exists(fn_histo):
                print(f"[WARN] Missing histology file: {fn_histo}")
                skipped += 1
                continue
        else:
            fn_histo = None

        emb_endo  = np.load(fn_endo)  if need_endo  else None
        emb_histo = np.load(fn_histo) if need_histo else None

        data.append((emb_endo, emb_histo, label, pat_id))

    labels = np.array([d[2] for d in data])

    print(f"[INFO] Target          : {target}")
    print(f"[INFO] Loaded samples  : {len(data)}")
    print(f"[INFO] Skipped samples : {skipped}")
    print(f"[INFO] Class 0 active  : {(labels == 0).sum()}")
    print(f"[INFO] Class 1 remission: {(labels == 1).sum()}")

    return data


def aggregate_patient_predictions(predictions):
    """
    Agrega predicciones por paciente usando worst-segment rule:
    prob_class1 del paciente = min(prob recto, prob sigma).
    """
    df = pd.DataFrame(predictions)

    agg_dict = {
        "true_label" : ("true_label",  "first"),
        "prob_class1": ("prob_class1", "min"),
        "n_segments" : ("prob_class1", "count"),
    }

    if "fold" in df.columns:
        agg_dict["folds"] = (
            "fold", lambda x: ",".join(map(str, sorted(set(x))))
        )

    patient_df = (
        df.groupby("pat_id")
        .agg(**agg_dict)
        .reset_index()
    )

    patient_df["pred_label"] = (patient_df["prob_class1"] >= 0.5).astype(int)

    return patient_df


def find_optimal_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_threshold = 0.5
    best_bal_acc   = -1.0

    for t in np.arange(0.01, 1.00, 0.01):
        y_pred = (y_prob >= t).astype(int)
        cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
        TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        sens = TP / max(TP + FN, 1)
        spec = TN / max(TN + FP, 1)
        bal  = (sens + spec) / 2
        if bal > best_bal_acc:
            best_bal_acc   = bal
            best_threshold = round(float(t), 2)

    return best_threshold, best_bal_acc


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    accuracy = (TP + TN) / max(TP + TN + FP + FN, 1)
    sensitivity = TP / max(TP + FN, 1)
    specificity = TN / max(TN + FP, 1)
    bal_acc = (sensitivity + specificity) / 2
    precision = TP / max(TP + FP, 1)
    npv = TN / max(TN + FN, 1)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    try:
        f1 = f1_score(y_true, y_pred, pos_label=1)
    except Exception:
        f1 = float("nan")

    return {
        "Accuracy" : round(float(accuracy),   4),
        "BAL_ACC" : round(float(bal_acc),     4),
        "Sensitivity" : round(float(sensitivity), 4),
        "Specificity" : round(float(specificity), 4),
        "Precision" : round(float(precision),   4),
        "NPV" : round(float(npv),         4),
        "F1" : round(float(f1),          4),
        "AUC" : round(float(auc),         4),
        "TP" : int(TP),
        "FN" : int(FN),
        "TN" : int(TN),
        "FP" : int(FP),
        "threshold" : round(float(threshold),   2),
        "confusion_matrix": cm,
    }


def plot_figures(
    train_acc_epoch,
    val_acc_epoch,
    train_loss_epoch,
    val_loss_epoch,
    train_bal_acc_epoch,
    val_bal_acc_epoch,
    run_name,
):
    os.makedirs("./local_data/results", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_epoch, label="Train loss")
    plt.plot(val_loss_epoch,   label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./local_data/results/loss_{run_name}.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_epoch,     label="Train accuracy")
    plt.plot(val_acc_epoch,       label="Val accuracy")
    plt.plot(train_bal_acc_epoch, label="Train balanced accuracy")
    plt.plot(val_bal_acc_epoch,   label="Val balanced accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./local_data/results/curves_{run_name}.png")
    plt.close()


def plot_confmx(conf_matrix, run_name, classes=(0, 1)):
    os.makedirs("./local_data/results", exist_ok=True)

    conf_matrix = np.asarray(conf_matrix)
    TP_diag     = np.diag(conf_matrix)
    FN_diag     = np.sum(conf_matrix, axis=1) - TP_diag

    with np.errstate(divide="ignore", invalid="ignore"):
        recalls = TP_diag / (TP_diag + FN_diag)
        recalls = np.nan_to_num(recalls)

    bal_acc = np.mean(recalls)

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        conf_matrix,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"BAL_ACC = {bal_acc:.4f}")
    plt.tight_layout()
    plt.savefig(f"./local_data/results/cfmx_{run_name}.png")
    plt.close()


def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False