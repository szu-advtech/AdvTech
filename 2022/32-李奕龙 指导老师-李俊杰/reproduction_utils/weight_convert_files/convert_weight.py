import torch
from collections import OrderedDict
import paddle
import json
import os


def convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path,
                                         paddle_dump_path,
                                         mapping_file,
                                         debug):
    print(">>> converting torch weight at ", pytorch_checkpoint_path)
    print(">>> to paddle weight at", paddle_dump_path)
    print(">>> based on layer mapping file ", mapping_file)
    if not os.path.exists(mapping_file):
        raise FileExistsError("run weight_mapping.py first to get torch to paddle mapping file")
    pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    paddle_state_dict = OrderedDict()
    paddle.set_device("cpu")
    with open(mapping_file, "r") as fp:
        torch2paddle = json.load(fp)

    for torch_layer_name, v in pytorch_state_dict.items():
        target_shape = torch2paddle[torch_layer_name]["target_shape"]
        if torch2paddle[torch_layer_name]["transpose"]:
            if target_shape[0] == target_shape[1] and debug:
                print(f"warning: transpose {torch_layer_name} with special shape")
            v = v.transpose(0, 1)

        assert target_shape == list(v.shape), \
            f"{torch_layer_name} to paddle {target_shape} not matched {list(v.shape)}"
        paddle_state_dict[torch2paddle[torch_layer_name]["paddle_layer"]] = v.data.numpy()
        if debug:
            if len(target_shape) == 1:
                print(f"torch {torch_layer_name}\n", v.data.numpy()[0])
                print(f"paddle {torch2paddle[torch_layer_name]['paddle_layer']}\n",
                      paddle_state_dict[torch2paddle[torch_layer_name]["paddle_layer"]][0])
            else:
                print(f"torch {torch_layer_name}\n", v.data.numpy()[0, :5])
                print(f"paddle {torch2paddle[torch_layer_name]['paddle_layer']}\n",
                      paddle_state_dict[torch2paddle[torch_layer_name]["paddle_layer"]][0, :5])
            print("===" * 10)
    paddle.save(paddle_state_dict, paddle_dump_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_checkpoint_path", type=str)
    parser.add_argument("--paddle_dump_path", type=str)
    parser.add_argument("--layer_mapping_file", type=str)
    parser.add_argument("--debug", type=bool,default=False)
    args = parser.parse_args()
    convert_pytorch_checkpoint_to_paddle(pytorch_checkpoint_path=args.pytorch_checkpoint_path,
                                         paddle_dump_path=args.paddle_dump_path,
                                         mapping_file=args.layer_mapping_file,
                                         debug=args.debug)
