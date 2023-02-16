from mvit.models import build_model
import argparse
import sys
import mvit.utils.checkpoint as cu
from mvit.config.defaults import assert_and_infer_cfg, get_cfg
import torchvision.transforms as transforms
import cv2


def parse_args():

    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="../configs/test/MVITv2_T_test.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See mvit/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    path = "../label"
    txt = open(path, "r", encoding="utf-8-sig")
    place = []
    for line in txt.readlines():
        line = line.strip()  # 去掉每行头尾空白
        if len(line) > 0:
            place.append(str(line))
    img = cv2.imread('../1.jpeg')
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    transf = transforms.ToTensor()
    img_tensor = transf(img)  # tensor数据格式是torch(C,H,W)
    img_tensor = img_tensor.unsqueeze(0)
    print(img_tensor.argmax())
    preds = model(img_tensor)
    print(place[preds.argmax().item()])

if __name__ == "__main__":
    main()