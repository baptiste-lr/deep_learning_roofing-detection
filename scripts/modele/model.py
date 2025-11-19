from torchvision.models import resnet18
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn as nn
import torchvision.models.segmentation as models
import segmentation_models_pytorch as smp

def get_unetpp_model(in_channels=3, num_classes=1, backbone="resnet34", encoder_weights=None):
    """
    Construit un modèle U-Net++ à partir de segmentation_models_pytorch.
    """
    model = smp.UnetPlusPlus(
        encoder_name=backbone,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes
    )
    return model

def get_deeplabv3_model(in_channels=3, num_classes=1, backbone="resnet50"):
    if backbone == "resnet18":
        backbone_model = resnet18(weights=None)
        # Modifier la 1ère couche si in_channels != 3
        if in_channels != 3:
            backbone_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Supprimer la couche de classification finale
        return_layers = {'layer4': 'out'}
        backbone = IntermediateLayerGetter(backbone_model, return_layers=return_layers)
        classifier = DeepLabHead(512, num_classes)
        model = models.DeepLabV3(backbone=backbone, classifier=classifier)

    elif backbone == "resnet50":
        model = models.deeplabv3_resnet50(weights=None)
        if in_channels != 3:
            model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    else:
        raise ValueError(f"Backbone '{backbone}' non supporté.")

    return model