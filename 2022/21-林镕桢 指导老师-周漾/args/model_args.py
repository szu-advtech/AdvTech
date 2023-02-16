import argparse

def get_model_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--is_encode", type=bool, default=False)
    parser.add_argument("--is_clip", type=bool, default=True)
    # parser.add_argument("--is_train", type=bool, default=True)

    # Generator opts
    parser.add_argument("--img_c", type=int, default=3)
    parser.add_argument("--z_c", type=int, default=256)
    parser.add_argument("--seg_c", type=int, default=184)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--upsample_times", type=int, default=6)
    parser.add_argument("--c_downsample_times", type=int, default=4)
    parser.add_argument("--last_spade_c", type=int, default=64)
    parser.add_argument("--spectral_norm", type=bool, default=True)
    parser.add_argument("--spade_hiden_c", type=int, default=128)

    # Discriminator args
    parser.add_argument("--start_c", type=int, default=64)
    parser.add_argument("--en_downsample_times", type=int, default=6)

    return parser.parse_args()