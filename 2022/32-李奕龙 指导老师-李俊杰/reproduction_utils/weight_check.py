import torch
from collections import OrderedDict
import paddle
from paddlenlp.transformers import CanineModel as PDmodel


def weight_check(pytorch_checkpoint_path,
                 paddle_dump_path,
                 mapping_file="./torch_paddle_layer_map.json"):
    # pytorch_state_dict = torch.load(pytorch_checkpoint_path, map_location="cpu")
    pd_model = PDmodel.from_pretrained('canine-s')
    paddle_state_dict = paddle.load(paddle_dump_path)
    loaded_state_dict = pd_model.state_dict()
    paddle.set_device("cpu")

    for k,v in paddle_state_dict.items():
        diff = paddle.mean(paddle.cast(v - loaded_state_dict[k],'float64')).numpy()[0]
        if abs(diff) > 1e-6:
            print(f"difference:\t {diff:.5f}")
            if v.ndim == 1:
                print(f"{k}\ntarget",v.numpy()[:5])
                print(f"loaded",loaded_state_dict[k].numpy()[:5])
            else:
                print(f"{k}\ntarget",v.numpy()[0,:5])
                print(f"loaded",loaded_state_dict[k].numpy()[0,:5])



if __name__ == "__main__":
    weight_check(pytorch_checkpoint_path="torch_weight/pytorch_model.bin",
                paddle_dump_path="../data/checkout_point/model_state.pdparams")