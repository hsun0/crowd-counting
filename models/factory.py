from .vgg_regressor import VGGCrowdRegressor
from .mobilenet_regressor import MobileNetCrowdRegressor
from .efficientnet_regressor import EfficientNetCrowdRegressor


def build_model(model_name: str):
    if model_name == "vgg":
        return VGGCrowdRegressor()
    elif model_name == "mobilenet":
        return MobileNetCrowdRegressor()
    elif model_name == "efficientnet":
        return EfficientNetCrowdRegressor()
    else:
        raise ValueError(f"Unknown model: {model_name}")
