import torchmetrics
import torch
from dataclasses import dataclass

class MulticlassMetrics():
    def __init__(self, args):
        self.args = args
        self.used_keys = {}
        self.num_classes = args.num_classes
        self.init_metrics()

    def init_metrics(self):
        # Initialize metrics for multiclass classification
        self.classifier_metrics_dict = {
            "acc": torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes).to(self.args.device),
            # "kappa": torchmetrics.CohenKappa(task='multiclass', num_classes=self.num_classes).to(self.args.device),
            "prec": torchmetrics.Precision(task='multiclass', num_classes=self.num_classes, average="macro").to(self.args.device),
            "recall": torchmetrics.Recall(task='multiclass', num_classes=self.num_classes, average="macro").to(self.args.device),
            "f1": torchmetrics.F1Score(task='multiclass', num_classes=self.num_classes, average="macro").to(self.args.device),
        }

    def fill_metrics(self, raw_predictions, raw_labels):
        # Convert raw predictions to probabilities and get predicted classes
        raw_pred = torch.softmax(raw_predictions, dim=1)
        predictions = torch.argmax(raw_pred, dim=1)
        labels = raw_labels

        # Update metrics
        self.classifier_metrics_dict["acc"].update(predictions, labels)
        # self.classifier_metrics_dict["kappa"].update(predictions, labels)
        self.classifier_metrics_dict["prec"].update(predictions, labels)
        self.classifier_metrics_dict["recall"].update(predictions, labels)
        self.classifier_metrics_dict["f1"].update(predictions, labels)

        self.used_keys = {
            "acc": True,
            "prec": True,
            "recall": True,
            "f1": True,
        }

    def compute_and_log_metrics(self, loss=0):
        metrics = {}
        for item in self.used_keys:
            metrics[item] = self.classifier_metrics_dict[item].compute()

        if loss != 0:
            metrics["loss_bce"] = loss

        return metrics

    def clear_metrics(self):
        for _, metric in self.classifier_metrics_dict.items():
            metric.reset()
        self.used_keys = {}

@dataclass
class MetricsArgs:
    num_classes: int
    device: str
