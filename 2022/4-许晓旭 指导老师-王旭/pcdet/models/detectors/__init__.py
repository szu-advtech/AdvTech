from .detector3d_template import Detector3DTemplate
from .PiFHNet import PiFHNet

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'PiFHNet': PiFHNet,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
