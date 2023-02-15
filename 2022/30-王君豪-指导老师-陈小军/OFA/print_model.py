
from fairseq import tasks
from fairseq import options
from utils import checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import utils
import argparse
parser=argparse.ArgumentParser
parser = options.get_generation_parser()

parser.add_argument("--ema-eval", action='store_true', help="Use EMA weights to make evaluation.")
parser.add_argument("--beam-search-vqa-eval", action='store_true',
                    help="Use beam search for vqa evaluation (faster inference speed but sub-optimal result), if not specified, we compute scores for each answer in the candidate set, which is slower but can obtain best result.")
parser.add_argument("--zero-shot", action='store_true')
args = options.parse_args_and_arch(parser)
cfg = convert_namespace_to_omegaconf(args)
overrides = eval(cfg.common_eval.model_overrides)
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths(cfg.common_eval.path),
    arg_overrides=overrides,
    suffix=cfg.checkpoint.checkpoint_suffix,
    strict=(cfg.checkpoint.checkpoint_shard_count == 1),
    num_shards=cfg.checkpoint.checkpoint_shard_count,
)

print(models)


