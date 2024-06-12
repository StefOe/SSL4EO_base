from lightly.utils.dist import print_rank_zero
from timm.models import create_model
from torch import nn
from torch import Tensor


def get_backbone(name: str, in_channels: int):
    try:
        model = create_model(name, pretrained=False, features_only=True)
    except RuntimeError:
        print_rank_zero(f"Could not find '{name}' backbone or it does not support 'features_only' mode, quitting now")
        quit()

    change_input_dims(model, in_channels)
    return model


def change_input_dims(model, in_channels):
    default_in_channels = 3

    # find modules with default inputs:
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            module.weight = nn.parameter.Parameter(
                Tensor(
                    module.out_channels,
                    in_channels // module.groups,
                    *module.kernel_size,
                )
            )
            module.reset_parameters()
        elif isinstance(module, nn.Linear) and module.in_features == default_in_channels:
            module.weight = nn.parameter.Parameter(
                Tensor(
                    module.out_features,
                    in_channels,
                )
            )
            module.reset_parameters()

    # only changing the obvious setting (there are more like "test_input_size" that are not always present)
    model.default_cfg["input_size"] = (in_channels, *model.default_cfg["input_size"][1:])
    return model
