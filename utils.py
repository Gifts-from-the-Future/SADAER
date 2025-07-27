import os
import queue
import shutil

import numpy as np
import torch.nn as nn
import torch
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,precision_recall_curve



def select_loss_fn(dataloader_name, device):
    if dataloader_name in ['ad', 'szclass']:
        return nn.CrossEntropyLoss().to(device)
    return nn.BCEWithLogitsLoss().to(device)


def find_optimal_threshold(y_true, y_scores):
    if len(np.unique(y_true)) > 2:
        raise NotImplementedError("Only binary classification is supported")
    precisions, recalls, thresh_vals = precision_recall_curve(y_true, y_scores)
    f1_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        denom = p + r
        if denom == 0:
            f1 = 0
        else:
            f1 = 2 * p * r / denom
        f1_scores.append(f1)
    best_index = np.argmax(f1_scores)
    best_threshold = thresh_vals[best_index]
    return best_threshold

def compute_metrics(predictions, targets, probabilities=None,avg_method='macro'):
    metrics = {}
    pred_map = defaultdict(list)
    true_map = defaultdict(list)
    if targets is not None:
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['f1'] = f1_score(targets, predictions, average=avg_method)
        metrics['precision'] = precision_score(targets, predictions, average=avg_method)
        metrics['recall'] = recall_score(targets, predictions, average=avg_method)
        if probabilities is not None:
            num_classes = len(set(targets))
            if num_classes <= 2:
                metrics['auroc'] = roc_auc_score(targets, probabilities)
            else:
                metrics['auroc'] = roc_auc_score(targets, probabilities, multi_class='ovr')
    return metrics, pred_map, true_map

def post_process_predictions(y_pred, y_true, y_prob, best_thresh, is_test, setName, dataloader_name):
    if setName == 'dev' and is_test and dataloader_name not in ['ad', 'szclass']:
        best_thresh = find_optimal_threshold(y_true=y_true, y_scores=y_prob)
        y_pred = (y_prob > best_thresh).astype(int)
    average_mode = 'binary' if dataloader_name not in ['ad', 'szclass'] else 'weighted'
    scores, _, _ = compute_metrics(
        predictions=y_pred,
        targets=y_true,
        probabilities=y_prob,
        avg_method=average_mode
    )
    return scores, best_thresh


class ModelCheckpointManager:
    def __init__(self, directory, monitor_metric, maximize=False, logger=None):
        self.directory = directory
        self.monitor_metric = monitor_metric
        self.maximize = maximize
        self.logger = logger
        self.best_metric = None
        self.saved_checkpoints = queue.PriorityQueue()
        os.makedirs(self.directory+'/model/', exist_ok=True)
        self._log(
            f'Initialized ModelCheckpointManager, monitoring to {"maximize" if maximize else "minimize"} "{monitor_metric}".')
    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    def _is_better(self, current_metric):
        if current_metric is None:
            return False
        if self.best_metric is None:
            return True
        if self.maximize:
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric
    def save_checkpoint(self, model, metric_value, filename):
        try:
            checkpoint = {'model_state_dict': model.state_dict()}
            filepath = os.path.join(self.directory+'/model/', filename)
            torch.save(checkpoint, filepath)
            self._log(f'Checkpoint saved at: {filepath}')
            if self._is_better(metric_value):
                self.best_metric = metric_value
                best_path = os.path.join(self.directory+'/best_model',filename)
                shutil.copyfile(filepath, best_path)
            return filepath
        except Exception as ex:
            self._log(f'Error during checkpoint save: {ex}')
            return None

def load_model_checkpoint(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self._total = 0.0
        self._count = 0
        self.avg = 0.0
    def update(self, value, n=1):
        self._total += value * n
        self._count += n
        self.avg = self._total / self._count if self._count != 0 else 0

def load_model(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'lstm' not in k.lower() and 'classifier' not in k.lower():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
    return new_state_dict

def load_gadn_weights_into_multitask_model(multi_model, gadn_model, task_id=0):
    multi_model.shared_conv1.load_state_dict(gadn_model.conv.state_dict(), strict=False)
    multi_model.shared_conv2.load_state_dict(gadn_model.conv2.state_dict(), strict=False)
    for i in range(len(gadn_model.encoder.dcgru_layers)):
        multi_model.multi_encoder.task_cells[task_id][i].load_state_dict(
            gadn_model.encoder.dcgru_layers[i].state_dict(), strict=False
        )
    multi_model.task_fc[task_id].load_state_dict(gadn_model.fc.state_dict(), strict=False)