import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.metrics import (
    classification_report,
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from tqdm import tqdm
from utils import AverageMeter

import seaborn as sns


def test(model, dataloader):
    model.eval()
    all_scores = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Test", dynamic_ncols=True, position=1):
            x = x.to(memory_format=torch.channels_last)
            output = model(x)

            scores = torch.softmax(output, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            all_scores.extend(predict)
            all_targets.extend(gt)
            all_probabilities.extend(scores)

    # Classification Report
    class_report = classification_report(
        all_targets, all_scores, labels=LABELS, target_names=NAMES, digits=4
    )
    print(f"Test process of all epoch is done. Classification Report:\n{class_report}")

    # Precision, Recall, F1
    precision = precision_score(
        all_targets, all_scores, labels=LABELS, average="macro", zero_division=0
    )
    recall = recall_score(
        all_targets, all_scores, labels=LABELS, average="macro", zero_division=0
    )
    f1 = f1_score(
        all_targets, all_scores, labels=LABELS, average="macro", zero_division=0
    )
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # ROC AUC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_probabilities = np.array(all_probabilities)
    assert len(LABELS) == len(NAMES), "Количество меток и названий классов не совпадает"

    # Расчет ROC AUC для каждого класса
    for i in range(len(LABELS)):
        # Убедитесь, что all_probabilities содержит вероятности для каждого класса
        fpr[i], tpr[i], _ = roc_curve(
            [1 if label == i else 0 for label in all_targets], all_probabilities[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    plt.figure(figsize=(12, 8))  # Установка размера графика
    for i in range(len(LABELS)):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"ROC curve of class {NAMES[i]} (area = {roc_auc[i]:.2f})",
        )
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC for each class")
    plt.legend(loc="lower right")
    roc_auc_fig = plt.gcf()
    plt.close()

    # PR AUC Curve
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(len(LABELS)):
        precision[i], recall[i], _ = precision_recall_curve(
            all_targets, all_probabilities[:, i], pos_label=i
        )
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure()
    plt.figure(figsize=(12, 8))  # Установка размера графика
    for i in range(len(LABELS)):
        plt.plot(
            recall[i],
            precision[i],
            label=f"PR curve of class {NAMES[i]} (area = {pr_auc[i]:.2f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve for each class")
    plt.legend(loc="lower left")
    pr_auc_fig = plt.gcf()
    plt.close()

    # Confusion Matrix
    
    cm = confusion_matrix(all_targets, all_scores, labels=LABELS)
    
    
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Создание тепловой карты
    plt.figure(figsize=(12, 12))

    plt.figure(figsize=(12, 12))
    # sns.heatmap(cm, annot=True, fmt="g", xticklabels=NAMES, yticklabels=NAMES)
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", xticklabels=NAMES, yticklabels=NAMES, cmap="YlGnBu")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    cm_fig = plt.gcf()
    plt.close()

    # Инициализация W&B (если еще не инициализировано)
    wandb.init(project="ваш_проект", entity="ваш_entity")

    # Загрузка графиков в W&B
    wandb.log({"ROC AUC": wandb.Image(roc_auc_fig)})
    wandb.log({"PR AUC": wandb.Image(pr_auc_fig)})
    wandb.log({"Confusion Matrix": wandb.Image(cm_fig)})

    return class_report, precision, recall, f1


{"12":"0", "9":"1","6":"2", "3": "3"}
LABELS = [0, 1, 2, 3]
NAMES = ["0", "1", "2", "3"]



def train(model, criterion, accelerator, train_loader, optimizer, epoch) -> None:
    model.train()
    loss_stat = AverageMeter("Loss")
    acc_stat = AverageMeter("Accuracy")

    train_iter = train_loader
    if accelerator.is_main_process:
        train_iter = tqdm(train_loader, desc="Train", dynamic_ncols=True, position=1)

    for step, (x, y) in enumerate(train_iter, start=1):
        optimizer.zero_grad(set_to_none=True)
        x = x.to(memory_format=torch.channels_last)

        output = model(x)
        loss = criterion(output, y)
        num_of_samples = x.shape[0]

        accelerator.backward(loss)
        loss_stat.update(loss.detach().cpu().item(), num_of_samples)

        scores = torch.softmax(output, dim=1).detach().cpu().numpy()
        predict = np.argmax(scores, axis=1)
        gt = y.detach().cpu().numpy()

        acc = np.mean(gt == predict)
        acc_stat.update(acc, num_of_samples)

        if accelerator.sync_gradients:
            accelerator.clip_grad_value_(model.parameters(), 0.2)

        optimizer.step()

        # if step % 100 and not step == 0:
        #     _, loss_avg = loss_stat()
        #     _, acc_avg = acc_stat()

        #     if accelerator.is_main_process:
        #         print(f"Epoch {epoch}, step: {step}: Loss: {loss_avg}, Acc: {acc_avg}")

    _, loss_avg = loss_stat()
    _, acc_avg = acc_stat()
    if accelerator.is_main_process:
        print(
            f"Train process of epoch {epoch} is done: Loss: {loss_avg}, Acc: {acc_avg}"
        )
    return loss_avg


def validation(model, criterion, dataloader, epoch) -> None:
    model.eval()
    loss_stat = AverageMeter("Loss")
    acc_stat = AverageMeter("Accuracy")
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Val", dynamic_ncols=True, position=1):
            x = x.to(memory_format=torch.channels_last)
            output = model(x)
            loss = criterion(output, y)
            num_of_samples = x.shape[0]

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)

            scores = torch.softmax(output, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            all_scores.extend(predict)
            all_targets.extend(gt)

            acc = np.mean(gt == predict)
            acc_stat.update(acc, num_of_samples)

        _, loss_avg = loss_stat()
        _, acc_avg = acc_stat()

        # Precision
        precision = precision_score(
            all_targets, all_scores, labels=LABELS, average="macro", zero_division=0
        )

        # Recall
        recall = recall_score(
            all_targets, all_scores, labels=LABELS, average="macro", zero_division=0
        )

        # F1 Score
        f1 = f1_score(
            all_targets, all_scores, labels=LABELS, average="macro", zero_division=0
        )

        print(
            f"Validation process of epoch {epoch} is done: Loss: {loss_avg}, Acc: {acc_avg}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}"
        )
        return precision, recall, f1, loss_avg
