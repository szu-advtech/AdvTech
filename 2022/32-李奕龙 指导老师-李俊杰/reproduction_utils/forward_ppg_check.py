import paddle
import torch
from canine import CanineTokenizer as PDTokenizer
from canine import CanineModel as PDmodel
from transformers import CanineTokenizer as PTTokenizer, \
    CanineModel as PTmodel
import random
import argparse
import time

text = [["a sae","123"]]

def timer(func):
    def wrapper(*args, **kwargs):
        time1 = time.time()
        result = func(*args, **kwargs)
        print(f"time cost for {func.__name__}: {time.time() - time1:.2f}s")
        return result
    return wrapper


def load_torch():
    PTtokenizer = PTTokenizer.from_pretrained("google/canine-s")
    pt_model = PTmodel.from_pretrained("google/canine-s")
    pt_model.eval()
    return pt_model, PTtokenizer


def load_paddle(model_name="canine-s"):
    PDtokenizer = PDTokenizer.from_pretrained(model_name)
    pd_model = PDmodel.from_pretrained(model_name)
    pd_model.eval()
    return pd_model, PDtokenizer


def keep_check(args):
    def random_sampling(batch_size=2):
        input_ids = []
        for _ in range(batch_size):
            text = ".".join([chr(random.randint(48, 1114112)) for _ in range(random.randint(4, 300))])
            input_ids.append(text)
        return input_ids
    print(">>> running forward propagation check, 10 random samples")

    pd_model, PDtokenizer = load_paddle(model_name=args.model_dir)
    pt_model, PTtokenizer = load_torch()
    for _ in range(args.num_test):
        acc_mean, acc_max = 0, 0
        text = random_sampling(2)
        pt_inputs = PTtokenizer(text, padding="longest", truncation=True, return_tensors="pt")
        inputs = PDtokenizer(text,
                             padding="longest",
                             return_attention_mask=True,
                             return_token_type_ids=True, )
        pd_inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}

        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs)
        with paddle.no_grad():
            outputs = pd_model(**pd_inputs)
            pd_outputs = [torch.from_numpy(outputs[i].numpy()) for i in range(len(outputs))]

        for i in range(len(pt_outputs)):
            acc_mean += torch.mean(torch.abs(pt_outputs[i] - pd_outputs[i]))
            acc_max += torch.max(torch.abs(pt_outputs[i] - pd_outputs[i]))
        print("mean diff:",acc_mean /len(pt_outputs), "max diff", acc_max / len(pt_outputs))
        if acc_max / len(pt_outputs) > 1e-3:
            print("diff to big")
            raise ValueError


def run_check(args):
    print(">>> running forward propagation check")
    @timer
    def get_torch_result():
        pt_model, PTtokenizer = load_torch()
        pt_inputs = PTtokenizer(text, padding="longest", truncation=True, return_tensors="pt")
        with torch.no_grad():
            pt_outputs = pt_model(**pt_inputs)
        return pt_outputs

    @timer
    def get_paddle_result():
        pd_model, PDtokenizer = load_paddle(model_name=args.model_dir)
        inputs = PDtokenizer(text,
                             padding="longest",
                             return_attention_mask=True,  # matters, need to add!!!
                             return_token_type_ids=True, )
        pd_inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}

        with paddle.no_grad():
            outputs = pd_model(**pd_inputs)
            outputs = [torch.from_numpy(outputs[i].numpy()) for i in range(len(outputs))]
        return outputs

    pt_outputs = get_torch_result()
    pd_outputs = get_paddle_result()
    print(">>> forward propagation result:")
    for i in range(len(pt_outputs)):
        print(f"mean difference:", torch.mean(torch.abs(pt_outputs[i] - pd_outputs[i])))
        print(f"max difference:", torch.max(torch.abs(pt_outputs[i] - pd_outputs[i])))
    if args.debug:
        print("==="*10)
        for i in range(len(pt_outputs)):
            for l in range(len(text)):
                print(f"batch {l} mean difference:", torch.mean(torch.abs(pt_outputs[i][l] - pd_outputs[i][l])))
                print(f"batch {l} max difference:", torch.max(torch.abs(pt_outputs[i][l] - pd_outputs[i][l])))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default="keep_check",type=str)
    parser.add_argument("--debug",action='store_true')
    parser.add_argument("--num_test", type=int, default=10, help="number of test")
    parser.add_argument("--model_dir",default="data/paddle_weight")
    return parser.parse_args()


def main():
    args = get_args()
    if args.mode == "keep_check":
        keep_check(args)
    else:
        run_check(args)


if __name__ == "__main__":
    main()