"""Module defining backbones."""
from models.backbones.backbone_base import BackBoneBase
from models.backbones.mobilenet import MobileNetEncoder
from models.backbones.shufflenet_v2 import ShuffleNetV2Encoder


str2enc = {"mobilenet": MobileNetEncoder, "shufllenetv2": ShuffleNetV2Encoder}

__all__ = ["BackBoneBase", "MobileNetEncoder", "ShuffleNetV2Encoder", "str2enc"]