import sys
import torch
from torch import nn
from typing import List

from src.denoising_diffusion_pytorch_big import Unet, GaussianDiffusion
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

def create_feature_extractor(model_type, **kwargs):
    """ Create the feature extractor for <model_type> architecture. """

    print("Creating DDPM Feature Extractor...")
    feature_extractor = FeatureExtractorDDPM(**kwargs)

    return feature_extractor

class FeatureExtractor(nn.Module):
    def __init__(self, model_path: str, input_activations: bool, **kwargs):
        '''
        Parent feature extractor class.

        param: model_path: path to the pretrained model
        param: input_activations:
            If True, features are input activations of the corresponding blocks
            If False, features are output activations of the corresponding blocks
        '''
        super().__init__()
        self._load_pretrained_model(model_path, **kwargs)
        print(f"Pretrained model is successfully loaded from {model_path}")

    def _load_pretrained_model(self, model_path: str, **kwargs):
        pass


class FeatureExtractorDDPM(FeatureExtractor):

    def __init__(self, steps: List[int], blocks: List[int], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps

    def _load_pretrained_model(self, model_path, **kwargs):
        import inspect
        self.model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8)
            # dim_mults=(1, 1, 2, 2, 4, 4)
        ).cpu()

        # Needed to pass only expected args to the function
        self.diffusion = GaussianDiffusion(
            self.model,
            image_size=(256, 256),
            timesteps=1000,  # number of steps
            loss_type='l2'  # L1 or L2
        ).cpu()
        ckp = torch.load('epoch=21-step=19249.ckpt', map_location='cpu')


        self.diffusion.load_state_dict(get_keys(ckp, 'diffusion'))


        self.model.to(device)
        self.diffusion.to(device)

        self.model.eval()

    @torch.no_grad()
    def forward(self, x, noise=None):

            for t in self.steps:
                # Compute x_t and run DDPM
                t = torch.tensor([t]).to(x.device)
                noisy_x = self.diffusion.q_sample(x, t, noise=noise)
                save=[]
#                final,save=self.model(noisy_x, self.diffusion._scale_timesteps(t),save)
                final,save=self.model(noisy_x,t,save)
                
                # Extract activations


            # Per-layer list of activations [N, C, H, W]
            return save


def collect_features(args, activations: List[torch.Tensor], sample_idx=0):
    """ Upsample activations and concatenate them to form a feature tensor """
    assert all([isinstance(acts, torch.Tensor) for acts in activations])
    size = tuple(args['dim'][:-1])
    resized_activations = []
    for feats in activations:
        feats = feats[sample_idx][None]
        feats = nn.functional.interpolate(
            feats, size=size, mode=args["upsample_mode"]
        )
        resized_activations.append(feats[0])

    return torch.cat(resized_activations, dim=0)
