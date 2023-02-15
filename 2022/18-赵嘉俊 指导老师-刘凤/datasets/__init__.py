import logging

from .base import IntractableLikelihoodError, DatasetNotAvailableError
from .spherical_simulator import SphericalGaussianSimulator
from .conditional_spherical_simulator import ConditionalSphericalGaussianSimulator
from .images import ImageNetLoader, CelebALoader, CardiacLoader, FFHQStyleGAN2DLoader, IMDBLoader, FFHQStyleGAN64DLoader
from .collider import WBFLoader, WBF2DLoader, WBF40DLoader
from .polynomial_surface_simulator import PolynomialSurfaceSimulator
from .lorenz import LorenzSimulator
from .utils import NumpyDataset

logger = logging.getLogger(__name__)


SIMULATORS = ["power", "spherical_gaussian", "conditional_spherical_gaussian", "lhc", "lhc40d", "lhc2d", "imagenet", "celeba", "cardiac", "gan2d", "gan64d", "lorenz", "imdb"]


def load_simulator(args):
    assert args.dataset in SIMULATORS
    if args.dataset == "power":
        simulator = PolynomialSurfaceSimulator(filename=args.dir + "/experiments/data/samples/power/manifold.npz")
    elif args.dataset == "spherical_gaussian":
        simulator = SphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "conditional_spherical_gaussian":
        simulator = ConditionalSphericalGaussianSimulator(args.truelatentdim, args.datadim, epsilon=args.epsilon)
    elif args.dataset == "lhc":
        simulator = WBFLoader()
    elif args.dataset == "lhc2d":
        simulator = WBF2DLoader()
    elif args.dataset == "lhc40d":
        simulator = WBF40DLoader()
    elif args.dataset == "imagenet":
        simulator = ImageNetLoader()
    elif args.dataset == "celeba":
        simulator = CelebALoader()
    elif args.dataset == "cardiac":
        simulator = CardiacLoader()
    elif args.dataset == "gan2d":
        simulator = FFHQStyleGAN2DLoader()
    elif args.dataset == "gan64d":
        simulator = FFHQStyleGAN64DLoader()
    elif args.dataset == "lorenz":
        simulator = LorenzSimulator()
    elif args.dataset == "imdb":
        simulator = IMDBLoader()
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    args.datadim = simulator.data_dim()
    return simulator
