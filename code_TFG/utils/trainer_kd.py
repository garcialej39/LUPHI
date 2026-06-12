"""
trainer_kd.py - Trainer para Knowledge Distillation MIL.

Loss total por muestra:
L = α · CE(student_logits, y_hard) + (1-α) · T² · KL(softmax(student/T) ‖ softmax(teacher/T))
"""

import copy
import numpy as np
import torch

from utils.my_utils import (
    plot_figures,
    aggregate_patient_predictions,
    compute_binary_metrics,
)

# Helpers

def _get_tensors(batch, device):
    emb_endo = torch.tensor(batch[0], dtype=torch.float32).to(device)
    emb_histo = torch.tensor(batch[1], dtype=torch.float32).to(device)
    label = torch.tensor(batch[2], dtype=torch.long).to(device)
    pat_id = batch[3]
    return emb_endo, emb_histo, label, pat_id


def _get_endo_tensor(batch, device):
    emb_endo = torch.tensor(batch[0], dtype=torch.float32).to(device)
    label = torch.tensor(batch[2], dtype=torch.long).to(device)
    pat_id = batch[3]
    return emb_endo, label, pat_id


# Train epoch
def train_epoch_kd(model, optimizer, criterion, train_data, device):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    predictions = []

    for i, batch in enumerate(train_data):
        optimizer.zero_grad()

        emb_endo, emb_histo, label, pat_id = _get_tensors(batch, device)

        student_logits, teacher_logits = model(emb_endo, emb_histo)

        loss, loss_ce, loss_kd = model.total_loss(
            student_logits, teacher_logits, label, criterion
        )

        if i == 0:
            print(f"  [DEBUG] loss_ce={loss_ce.item():.4f} | "
                  f"loss_kd={loss_kd.item():.4f} | "
                  f"loss_total={loss.item():.4f}")

        loss.backward()
        optimizer.step()

        probs = torch.softmax(student_logits.detach(), dim=0)
        prob_class1 = probs[1].item()
        pred = torch.argmax(probs).item()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_kd  += loss_kd.item()

        predictions.append({
            "pat_id" : pat_id,
            "true_label" : int(label.item()),
            "pred_label" : int(pred),
            "prob_class1": float(prob_class1),
        })

    patient_preds = aggregate_patient_predictions(predictions)
    metrics = compute_binary_metrics(
        y_true=patient_preds["true_label"].values,
        y_prob=patient_preds["prob_class1"].values,
        threshold=0.5,
    )

    n = len(train_data)
    return total_loss / n, total_ce / n, total_kd / n, metrics

# Evaluate epoch (solo student)
def evaluate_epoch_kd(model, data, criterion, device):
    model.eval()

    total_loss = 0.0
    predictions = []

    with torch.no_grad():
        for batch in data:
            emb_endo, label, pat_id = _get_endo_tensor(batch, device)

            student_logits, _ = model(emb_endo, features_histo=None)

            loss = criterion(
                student_logits.unsqueeze(0),
                label.unsqueeze(0),
            )

            probs = torch.softmax(student_logits, dim=0)
            prob_class1 = probs[1].item()
            pred = torch.argmax(probs).item()

            total_loss += loss.item()

            predictions.append({
                "pat_id" : pat_id,
                "true_label" : int(label.item()),
                "pred_label" : int(pred),
                "prob_class1": float(prob_class1),
            })

    patient_preds = aggregate_patient_predictions(predictions)
    metrics = compute_binary_metrics(
        y_true=patient_preds["true_label"].values,
        y_prob=patient_preds["prob_class1"].values,
        threshold=0.5,
    )

    return total_loss / len(data), metrics

# Train model
def train_model_kd(
    model,
    optimizer,
    criterion,
    scheduler,
    train_data,
    val_data,
    epochs,
    run_name,
    device,
):
    train_acc_epoch  = []
    val_acc_epoch = []
    train_bal_acc_epoch = []
    val_bal_acc_epoch = []
    train_loss_epoch  = []
    val_loss_epoch = []

    best_val_auc = -1.0
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        indices          = np.random.permutation(len(train_data))
        train_data_epoch = [train_data[i] for i in indices]

        train_loss, ce_loss, kd_loss, train_metrics = train_epoch_kd(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_data=train_data_epoch,
            device=device,
        )

        val_loss, val_metrics = evaluate_epoch_kd(
            model=model,
            data=val_data,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        train_acc_epoch.append(train_metrics["Accuracy"])
        val_acc_epoch.append(val_metrics["Accuracy"])
        train_bal_acc_epoch.append(train_metrics["BAL_ACC"])
        val_bal_acc_epoch.append(val_metrics["BAL_ACC"])
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)

        print("-" * 80)
        print(f"Epoch {epoch + 1:03d}/{epochs}")
        print(f"Train | Loss={train_loss:.4f} (CE={ce_loss:.4f} KD={kd_loss:.4f}) | "
              f"BAL_ACC={train_metrics['BAL_ACC']:.4f} | AUC={train_metrics['AUC']:.4f}")
        print(f"Val   | Loss={val_loss:.4f} | "
              f"BAL_ACC={val_metrics['BAL_ACC']:.4f} | AUC={val_metrics['AUC']:.4f}")
        print("-" * 80)

        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    print(f"[INFO] Best checkpoint: epoch={best_epoch}, val_AUC={best_val_auc:.4f}")
    
    plot_figures(
        train_acc_epoch=train_acc_epoch,
        val_acc_epoch=val_acc_epoch,
        train_loss_epoch=train_loss_epoch,
        val_loss_epoch=val_loss_epoch,
        train_bal_acc_epoch=train_bal_acc_epoch,
        val_bal_acc_epoch=val_bal_acc_epoch,
        run_name=run_name,
    )

# Validate (inferencia — solo endoscopia)
def validate_model_kd(model, val_data, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in val_data:
            emb_endo, label, pat_id = _get_endo_tensor(batch, device)
            student_logits, _ = model(emb_endo, features_histo=None)
            probs             = torch.softmax(student_logits, dim=0)
            prob_class1       = probs[1].item()
            pred              = torch.argmax(probs).item()

            predictions.append({
                "pat_id" : pat_id,
                "true_label" : int(label.item()),
                "pred_label" : int(pred),
                "prob_class1": float(prob_class1),
            })

    return predictions