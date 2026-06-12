"""
trainer.py - Bucle de entrenamiento y evaluación para modelos MIL unimodales y multimodales
-Las predicciones se agregan a nivel de paciente antes de calcular métricas.
-Compatible con modalidades: 'endo_only', 'histo_only', 'multimodal'.
"""
import copy
import numpy as np
import torch

from utils.my_utils import (
    plot_figures,
    aggregate_patient_predictions,
    compute_binary_metrics,
)


def get_batch_tensor(batch, data_modality, device):
    if data_modality == "endo_only":
        return [torch.tensor(batch[0], dtype=torch.float32).to(device)]
    elif data_modality == "histo_only":
        return [torch.tensor(batch[1], dtype=torch.float32).to(device)]
    else:
        return [
            torch.tensor(batch[0], dtype=torch.float32).to(device),
            torch.tensor(batch[1], dtype=torch.float32).to(device),
        ]


def evaluate_epoch(model, data, labels, criterion, data_modality, device):
    model.eval()
    total_loss  = 0.0
    predictions = []

    with torch.no_grad():
        for i, batch in enumerate(data):
            batch_tensor = get_batch_tensor(batch, data_modality, device)
            logits = model(batch_tensor)
            label_tensor = torch.tensor(labels[i], dtype=torch.long).to(device)
            loss = criterion(logits.unsqueeze(0), label_tensor.unsqueeze(0))

            probs = torch.softmax(logits, dim=0)
            pred = torch.argmax(probs).item()
            prob_class1 = probs[1].item()
            total_loss += loss.item()

            predictions.append({
                "pat_id"     : batch[3],
                "true_label" : int(labels[i]),
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


def train_epoch(model, optimizer, criterion, train_data, train_labels, data_modality, device):
    model.train()
    total_loss  = 0.0
    predictions = []

    for i, batch in enumerate(train_data):
        optimizer.zero_grad()
        batch_tensor = get_batch_tensor(batch, data_modality, device)
        logits = model(batch_tensor)
        label_tensor = torch.tensor(train_labels[i], dtype=torch.long).to(device)
        loss = criterion(logits.unsqueeze(0), label_tensor.unsqueeze(0))
        loss.backward()
        optimizer.step()

        probs = torch.softmax(logits.detach(), dim=0)
        pred = torch.argmax(probs).item()
        prob_class1 = probs[1].item()
        total_loss += loss.item()

        predictions.append({
            "pat_id"     : batch[3],
            "true_label" : int(train_labels[i]),
            "pred_label" : int(pred),
            "prob_class1": float(prob_class1),
        })

    patient_preds = aggregate_patient_predictions(predictions)
    metrics = compute_binary_metrics(
        y_true=patient_preds["true_label"].values,
        y_prob=patient_preds["prob_class1"].values,
        threshold=0.5,
    )
    return total_loss / len(train_data), metrics


def train_model(
    model, optimizer, criterion, scheduler,
    train_data, train_labels, val_data, val_labels,
    epochs, run_name, data_modality, device,
):
    train_acc_epoch = []
    val_acc_epoch = []
    train_bal_acc_epoch = []
    val_bal_acc_epoch = []
    train_loss_epoch = []
    val_loss_epoch = []

    best_val_auc = -1.0   # ← criterio AUC (consistente con trainer_kd)
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        indices = np.random.permutation(len(train_labels))
        train_data_epoch = [train_data[idx] for idx in indices]
        train_labels_epoch = train_labels[indices]

        train_loss, train_metrics = train_epoch(
            model=model, optimizer=optimizer, criterion=criterion,
            train_data=train_data_epoch, train_labels=train_labels_epoch,
            data_modality=data_modality, device=device,
        )
        val_loss, val_metrics = evaluate_epoch(
            model=model, data=val_data, labels=val_labels,
            criterion=criterion, data_modality=data_modality, device=device,
        )

        scheduler.step()

        train_acc_epoch.append(train_metrics["Accuracy"])
        val_acc_epoch.append(val_metrics["Accuracy"])
        train_bal_acc_epoch.append(train_metrics["BAL_ACC"])
        val_bal_acc_epoch.append(val_metrics["BAL_ACC"])
        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)

        print("-" * 80)
        print(f"Epoch {epoch+1:03d}/{epochs}")
        print(f"Train | Loss={train_loss:.4f} | ACC={train_metrics['Accuracy']:.4f} | "
              f"BAL_ACC={train_metrics['BAL_ACC']:.4f} | AUC={train_metrics['AUC']:.4f}")
        print(f"Val   | Loss={val_loss:.4f} | ACC={val_metrics['Accuracy']:.4f} | "
              f"BAL_ACC={val_metrics['BAL_ACC']:.4f} | AUC={val_metrics['AUC']:.4f}")
        print("-" * 80)

        if val_metrics["AUC"] > best_val_auc:
            best_val_auc = val_metrics["AUC"]
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    print(f"[INFO] Best checkpoint: epoch={best_epoch}, val_AUC={best_val_auc:.4f}")

    plot_figures(
        train_acc_epoch=train_acc_epoch, val_acc_epoch=val_acc_epoch,
        train_loss_epoch=train_loss_epoch, val_loss_epoch=val_loss_epoch,
        train_bal_acc_epoch=train_bal_acc_epoch, val_bal_acc_epoch=val_bal_acc_epoch,
        run_name=run_name,
    )


def validate_model(model, val_data, val_labels, data_modality, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for i, batch in enumerate(val_data):
            batch_tensor = get_batch_tensor(batch, data_modality, device)
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=0)
            pred = torch.argmax(probs).item()
            prob_class1 = probs[1].item()

            predictions.append({
                "pat_id" : batch[3],
                "true_label" : int(val_labels[i]),
                "pred_label" : int(pred),
                "prob_class1": float(prob_class1),
            })

    return predictions