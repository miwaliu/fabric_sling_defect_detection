import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from utils.metrics import get_metrics_values
from utils.util import AverageMeter


def validate_model(
        epoch: int, model, data_loader, criterion, best_val_iou: float, n_classes: int, output_dir: str):
    """
    Валидацця модели

    :param epoch (int): номер текущей эпохи
    :param model: Модель
    :param data_loader: Pytorch DataLoader
    :param criterion: Функция потерь
    :param best_val_iou (float): Текущее лучшее значение параметра ntersection over Union
    :param n_classes (int): Колличество классов
    :param output_dir (str): Путь по которому будут сохранены веса
    :return: Значение параметра Intersection over Union
    """

    model.eval()
    losses = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            inputs = sample["image"].type(torch.FloatTensor)
            targets = sample["label"].type(torch.LongTensor)
            if torch.cuda.is_available():
                targets = targets.cuda()
                inputs = inputs.cuda()

            outputs = model(inputs, (targets.shape[1], targets.shape[2]))["output"]
            if (
                    outputs.shape[1] != targets.shape[1]
                    or outputs.shape[2] != targets.shape[2]
            ):
                outputs = F.upsample(
                    input=outputs,
                    size=(targets.shape[1], targets.shape[2]),
                    mode="bilinear",
                )

            loss = criterion(outputs, targets)

            inters, uni = get_metrics_values(targets, [outputs], n_classes)
            intersection_meter.update(inters)
            union_meter.update(uni)

            losses.update(loss.item())
            sys.stdout.write(
                "\r[Epoch %d] [Batch %d / %d]" % (epoch, i, len(data_loader))
            )

    iou = intersection_meter.sum / union_meter.sum

    if best_val_iou < np.mean(iou):
        best_val_iou = np.mean(iou)
        torch.save(
            {"state_dict": model.state_dict(), "iou": np.mean(iou), "epoch": epoch},
            os.path.join(output_dir, "best.pth"),
        )

    for i, _iou in enumerate(iou):
        print("\nclass [{}], IoU: {:.4f}".format(i, _iou))

    print("Epoch Loss", losses.avg, "Validation_meanIou", np.mean(iou))
    return best_val_iou
