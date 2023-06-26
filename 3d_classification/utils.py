import monai.networks.nets as nets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch.nn.functional as F


def draw_confusion_graph(y_pred, y, directory):
    y_pred = y_pred.argmax(dim=1)
    y = y.argmax(dim=1)

    print(y.cpu().numpy())
    print(y_pred.cpu().numpy())

    cm = confusion_matrix(
        y.cpu().numpy(),
        y_pred.cpu().numpy(),
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["B", "M"],
    )
    fig, ax = plt.subplots(1, 1, facecolor='white')
    _ = disp.plot(ax=ax)
    plt.savefig(directory)
    plt.close(fig)


class SmoothCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(SmoothCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)

        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))

        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def get_model(args):
    if args.model == 'densenet':
        model = nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2, dropout_prob=args.dropout)
    elif args.model == 'resnet':
        model = nets.resnet152(spatial_dims=3, n_input_channels=1, num_classes=2)
    elif args.model == 'unet':
        model = nets.UNet(spatial_dims=3, in_channels=1, out_channels=2, dropout=args.dropout)
    elif args.model == 'efficientnet':
        model = nets.EfficientNet(spatial_dims=3, in_channels=1, out_channels=2, dropout=args.dropout)
