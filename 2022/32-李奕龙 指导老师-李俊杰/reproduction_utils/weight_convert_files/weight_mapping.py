from transformers import CanineModel as PTmodel
from transformers import CanineConfig as PTconfig
import json
from canine import CanineModel as PDmodel
import re


def run_check():
    layer_mapping = {}
    config = PTconfig.from_pretrained('google/canine-s')
    ptg_model = PTmodel(config=config)

    tw = ptg_model.state_dict()
    tw_list = list(ptg_model.state_dict().keys())

    pdg_model = PDmodel()
    pw = pdg_model.state_dict()
    pw_list = list(pw.keys())

    count = 0
    max_len = max([len(x) for x in tw_list])

    for torch_layer in tw_list.copy():
        prefix = ""
        if torch_layer.startswith("encoder"):
            mapping = encoder_mapping
            # prefix = "main_"
        elif torch_layer.startswith("final_char_encoder") or torch_layer.startswith("initial_char_encoder"):
            mapping = other_encoder_mapping
        elif torch_layer.startswith("pooler"):
            mapping = pooler_mapping
        else:
            mapping = remaining_mapping

        for huggingface_name, paddle_name in mapping.items():
            paddle_layer = prefix + re.sub(huggingface_name, paddle_name, torch_layer)
            if paddle_layer in pw_list:
                transpose = False
                pd_shape =  list(pw[paddle_layer].shape)
                if torch_layer[-7:] == ".weight":
                    if not any([w in torch_layer for w in dont_transpose]):
                        if tw[torch_layer].ndim == 2:
                            tw[torch_layer] = tw[torch_layer].transpose(0, 1)
                            transpose = True
                            # elif not torch_layer.startswith("encoder"):
                            #     tw[torch_layer] = tw[torch_layer].transpose(0, 1)
                            #     transpose = True

                pt_shape = list(tw[torch_layer].shape)
                if transpose and pd_shape[0] == pd_shape[1]:
                    print(f"warning: {paddle_layer} transpose with special shape")
                if pt_shape == pd_shape:
                # 打印torch与paddle对应的权重形状，可以确认该权重转换的时候是否需要转置。
                    count += 1
                    layer_mapping[torch_layer] = {"paddle_layer": paddle_layer,
                                                  "transpose": transpose,
                                                  "target_shape": pd_shape
                                                  }
                    # print(k + " " * (max_len - len(k)), "\t", pt_shape, "\t", pd_shape, "\t", new_k)
                    pw_list.remove(paddle_layer)
                    tw_list.remove(torch_layer)

                else:
                    print("*************shape error**************")
                    print(torch_layer + " " * (max_len - len(torch_layer)), "\t", pt_shape, "\t", pd_shape, "\t", paddle_layer)
                    print("**************************")
    if len(pw_list) + len(tw_list) == 0:
        print("all layer matched, storing mapping files...")
        with open("torch_paddle_layer_map.json", "w") as fp:
            fp.write(json.dumps(layer_mapping,indent=4))
    else:
        print("==="*20)
        print("list of weights no matched:")
        for i in pw_list:
            print(i)
        print("==="*20)
        for i in tw_list:
            print(i)
    print("num matched: ", count)


encoder_mapping = {
    r"layer.(\d+).attention.self.query":r"layers.\1.attention.self_attn.q_proj",
    r"layer.(\d+).attention.self.key": r"layers.\1.attention.self_attn.k_proj",
    r"layer.(\d+).attention.self.value": r"layers.\1.attention.self_attn.v_proj",

    r"layer.(\d+)(.\w+)?.output.LayerNorm": r"layers.\1\2.layer_norm",
    r"layer.(\d+).output.dense":r"layers.\1.ffn_output",
    r"layer.(\d+).intermediate.dense": r"layers.\1.ffn",
    r"layer.(\d+).attention.output.(\w+)":r"layers.\1.attention.\2"
}

other_encoder_mapping = {
    r"layer.(\d+).attention.self.query": r"layers.\1.attention.self_attn.q_proj",
    r"layer.(\d+).attention.self.key": r"layers.\1.attention.self_attn.k_proj",
    r"layer.(\d+).attention.self.value": r"layers.\1.attention.self_attn.v_proj",

    r"layer.(\d+)(.\w+)?.output.LayerNorm": r"layers.\1\2.layer_norm",
    r"layer.(\d+).output.dense": r"layers.\1.ffn_output",
    r"layer.(\d+).intermediate.dense": r"layers.\1.ffn",
    r"layer.(\d+).attention.output.(\w+)": r"layers.\1.attention.\2"
}

remaining_mapping = {
    r"":r"",
    r"LayerNorm":r"layer_norm"
}

pooler_mapping = {
    r".dense":r""
}

dont_transpose = [
  "HashBucketCodepointEmbedder","_position_embeddings","_type_embeddings",
    "_embeddings.weight", "_layer_norm",
    "embed_positions", "embed_tokens", "layernorm_embedding",
    "lm_head", "shared",
    # "attention"
]

if __name__ == "__main__":
    run_check()