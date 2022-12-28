import os
import logging
import numpy as np
import torch
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def create_filename(type_, label, args):
    run_label = "_run{}".format(args.i) if args.i > 0 else ""

    if type_ == "dataset":  # Fixed datasets
        filename = "data/samples/{}".format(args.dataset)
        # filename = "{}/experiments/data/samples/{}".format(args.dir, args.dataset)

    elif type_ == "sample":  # Dynamically sampled from simulator
        filename = "data/samples/{}/{}{}.npy".format(args.dataset, label, run_label)

    elif type_ == "model":
        filename = "data/models/{}.pt".format(args.modelname)

    elif type_ == "checkpoint":
        filename = "data/models/checkpoints/{}_{}_{}.pt".format(args.modelname, "epoch" if label is None else "epoch_" + label, "{}")

    elif type_ == "resume":
        for label in ["D_", "C_", "B_", "A_", ""]:
            filename = "data/models/checkpoints/{}_epoch_{}last.pt".format(args.modelname, label, "last")
            if os.path.exists(filename):
                return filename

        raise FileNotFoundError(f"Trying to resume training from {filename}, but file does not exist")

    elif type_ == "training_plot":
        filename = "figures/training/{}_{}_{}.pdf".format(args.modelname, "epoch" if label is None else label, "{}")

    elif type_ == "learning_curve":
        filename = "data/learning_curves/{}.npy".format(args.modelname)

    elif type_ == "results":
        trueparam_name = "" if args.trueparam is None or args.trueparam == 0 else "_trueparam{}".format(args.trueparam)
        filename = "data/results/{}_{}{}.npy".format(args.modelname, label, trueparam_name)

    elif type_ == "mcmcresults":
        trueparam_name = "" if args.trueparam is None or args.trueparam == 0 else "_trueparam{}".format(args.trueparam)
        chain_name = "_chain{}".format(args.chain) if args.chain > 0 else ""
        filename = "data/results/{}_{}{}{}.npy".format(args.modelname, label, trueparam_name, chain_name)

    elif type_ == "timing":
        filename = "data/timing/{}_{}_{}_{}_{}_{}{}.npy".format(
            args.algorithm,
            args.outerlayers,
            args.outertransform,
            "mlp" if args.outercouplingmlp else "resnet",
            args.outercouplinglayers,
            args.outercouplinghidden,
            run_label,
        )
    elif type_ == "paramscan":
        filename = "data/paramscan/{}.pickle".format(args.paramscanstudyname)
    else:
        raise NotImplementedError

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename


def create_modelname(args):
    # 格式化字符串的函数str.format():
    # 通过位置映射：'{1},{0}'.format('kzc',18)->'18,kzc'
    # 通过关键字映射：'{age},{name}'.format(name='kzc',age=18)->'18,kzc'
    # 通过下标映射：'{0[1]},{0[0]}'.format(['kzc',18])->'18,kzc'
    # 使用格式限定符（语法是{}中带:号），其中^、<、>分别是居中、左对齐、右对齐，后面带宽度，:号后面带填充的字符，只能是一个字符，不指定的话默认是用空格填充
    # '{:a>8}'.format('189')->'aaaaa189'#
    run_label = "_run{}".format(args.i) if args.i > 0 else ""
    appendix = "" if args.modelname is None else "_" + args.modelname

    try:
        if args.truth:
            if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
                args.modelname = "truth_{}_{}_{}_{:.3f}{}{}".format(args.dataset, args.truelatentdim, args.datadim, args.epsilon, appendix, run_label)
            else:
                args.modelname = "truth_{}{}{}".format(args.dataset, appendix, run_label)
            return
    except:
        pass

    if args.dataset in ["spherical_gaussian", "conditional_spherical_gaussian"]:
        args.modelname = "{}{}_{}_{}_{}_{}_{:.3f}{}{}".format(
            args.algorithm, "_specified" if args.specified else "", args.modellatentdim, args.dataset, args.truelatentdim, args.datadim, args.epsilon, appendix, run_label,
        )
    else:
        args.modelname = "{}{}_{}_{}{}{}".format(args.algorithm, "_specified" if args.specified else "", args.modellatentdim, args.dataset, appendix, run_label)


def nat_to_bit_per_dim(dim):
    if isinstance(dim, (tuple, list, np.ndarray)):
        dim = np.product(dim)
    logger.debug("Nat to bit per dim: factor %s", 1.0 / (np.log(2) * dim))
    return 1.0 / (np.log(2) * dim)


def sum_except_batch(x, num_batch_dims=1):
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def array_to_image_folder(data, folder):
    for i, x in enumerate(data):
        x = np.clip(np.transpose(x, [1, 2, 0]) / 256.0, 0.0, 1.0)
        if i == 0:
            logger.debug("x: %s", x)
        plt.imsave(f"{folder}/{i}.jpg", x)
