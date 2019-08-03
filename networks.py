import torch
import torch.nn as nn
import torchvision.models as models


def resnet18(num_classes=100):
    """Constructs a ResNet-18 model for CIFAR dataset"""
    model = models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.avgpool = nn.AvgPool2d(4, stride=1)
    model.maxpool = nn.Identity()
    return model


def vgg11(num_classes=100):
    """Constructs a VGG-11 model for CIFAR dataset"""
    model = models.vgg11_bn(num_classes=num_classes)
    model.avgpool = nn.Identity()
    model.classifier[0] = nn.Linear(512, 4096)
    return model


def vgg11s(num_classes=100):
    """Constructs a VGG-11 simplified model for CIFAR dataset"""
    model = models.vgg11_bn()
    model.avgpool = nn.Identity()
    model.classifier = nn.Linear(512, num_classes)
    return model


def densenet63(num_classes=100):
    """Constructs a DenseNet-63 simplified model for CIFAR dataset"""
    num_init_features = 32
    model = models.densenet._densenet('densenet63', 32, (3, 6, 12, 8), num_init_features, pretrained=False, progress=False)
    model.features[0] = nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[1] = nn.BatchNorm2d(num_init_features)
    model.features[3] = nn.Identity()
    model.classifier = nn.Linear(512, num_classes)
    return model


if __name__ == "__main__":
    from torchsummary import summary
    model = densenet63(num_classes=100)
    summary(model, (3, 32, 32), device="cpu")
    print(model)
