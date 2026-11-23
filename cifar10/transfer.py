import torch
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def resnet_transforms():
    eval_transform = WEIGHTS.transforms()
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            eval_transform,
        ]
    )
    return train_transform, eval_transform


class FrozenResNet18(nn.Module):
    def __init__(self, num_classes=10, *, weights=WEIGHTS):
        super().__init__()
        self.encoder = resnet18(weights=weights)
        feature_count = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.head = nn.Linear(feature_count, num_classes)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        return self.encoder(images)

    def forward(self, images):
        return self.head(self.encode(images))


class Layer4ResNet18(FrozenResNet18):
    """Train layer4 and the head while keeping encoder BatchNorm statistics fixed."""

    def __init__(self, num_classes=10, *, weights=WEIGHTS):
        super().__init__(num_classes=num_classes, weights=weights)
        self.encoder.layer4.requires_grad_(True)

    def encode(self, images):
        # The frozen-head implementation disabled autograd for the whole encoder.
        # Here autograd must track operations involving the trainable layer4 tensors.
        return self.encoder(images)
