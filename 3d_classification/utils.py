import monai.networks.nets as nets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch.nn.functional as F
import torch



def draw_confusion_graph(y_pred, y, directory):
    y_pred = y_pred.argmax(dim=1)
    y = y.argmax(dim=1)

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
    def __init__(self, label_smoothing=0.0, alpha=1., gamma=2.):
        super().__init__()
        self.label_smoothing = label_smoothing

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):

        logsoftmax = torch.nn.LogSoftmax(dim=1)

        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        if self.label_smoothing > 0.0:
            s_by_c = self.label_smoothing / len(input[0])
            smooth = torch.zeros_like(target)
            smooth = smooth + s_by_c
            target = target * (1. - s_by_c) + smooth

        cross_entropy_loss = torch.sum(-target * logsoftmax(input), dim=1)
        pt = torch.exp(-cross_entropy_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * cross_entropy_loss

        return F_loss.mean()

class FocalLossCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):

        logsoftmax = torch.nn.LogSoftmax(dim=1)

        if len(target.size()) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()

        cross_entropy_loss = torch.sum(-target * logsoftmax(input), dim=1)
        pt = torch.exp(-cross_entropy_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * cross_entropy_loss

        return F_loss.mean()

def get_model(args):
    if args.model == 'densenet':
        model = nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2, dropout_prob=args.dropout)
    elif args.model == 'resnet':
        # model = nets.resnet101(spatial_dims=3, n_input_channels=1, num_classes=2, pretrained=False)
        model = resnet101(num_seg_classes=2, sample_input_W=28, sample_input_H=28, sample_input_D=14)
        weights = torch.load('resnet_50.pth')
        model.load_state_dict(weights)

    elif args.model == 'unet':
        model = nets.UNet(spatial_dims=3, in_channels=1, out_channels=2, dropout=args.dropout)
    elif args.model == 'efficientnet':
        model = nets.EfficientNet(spatial_dims=3, in_channels=1, out_channels=2, dropout=args.dropout)

    return model
