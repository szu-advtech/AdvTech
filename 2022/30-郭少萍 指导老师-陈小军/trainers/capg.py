import os
import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from transformers import AutoConfig

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from transformers.models.roberta.modeling_roberta import (
        RobertaLayer,
        RobertaSelfAttention,
        RobertaAttention,
        RobertaSelfOutput,
        RobertaIntermediate,
        RobertaOutput,
        )

from trainers.imagenet_templates import IMAGENET_TEMPLATES

# 根据backnone name下载模型至本地并加载模型
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)  # (n_cls, L, ctx_dim)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # text_projection
        # x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].shape : [n_cls, ctx_dim]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


# 分词器
_tokenizer = _Tokenizer()


class SPSelfAttention(RobertaSelfAttention):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        hidden_states,
        word_embed_lists,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(word_embed_lists))
            value_layer = self.transpose_for_scores(self.value(word_embed_lists))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(word_embed_lists))
            value_layer = self.transpose_for_scores(self.value(word_embed_lists))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class SPAttention(RobertaAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = SPSelfAttention(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        word_embed_lists,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            word_embed_lists,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SPLayer(RobertaLayer):
    def __init__(self, config):
        super().__init__(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SPAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = SPAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        hidden_states,
        word_embed_lists,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            word_embed_lists,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        #layer_output = apply_chunking_to_forward(
        #    self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        #)
        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        outputs = outputs[0]
        return outputs


class Adapter(nn.Module):
    def __init__(self, hidden_size, prompt_len):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(1, hidden_size)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(hidden_size, prompt_len))
        ]))

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [N, D , 1]
        x = self.adapter(x)
        x = x.permute(0, 2, 1)  # [N, prompt_len, D]
        return x


def reset_SPLayer_parameters(clip_model, pg):
    clip_state_dict = clip_model.transformer.resblocks[0].state_dict()
    pg_state_dict = pg.state_dict()
    reset_state_dict = pg_state_dict

    clip_keys = []
    clip_values = []
    for key, value in clip_state_dict.items():
        clip_keys.append(key)
        clip_values.append(value)

    pg_keys = []
    for key, value in pg_state_dict.items():
        pg_keys.append(key)

    pg_qkv_w = torch.chunk(clip_values[0], 3, dim=0)
    pg_qkv_bias = torch.chunk(clip_values[1], 3, dim=0)

    for i in range(3):
        reset_state_dict[pg_keys[2*i]] = pg_qkv_w[i]
        reset_state_dict[pg_keys[2*i+1]] = pg_qkv_bias[i]

    for i in range(6, len(pg_state_dict)):
        reset_state_dict[pg_keys[i]] = clip_values[i-4]

    pg.load_state_dict(reset_state_dict)


# 只需改写 PromptLearner
# 输入：image feature改写的embedding, prompt_word_sets的embedding, clip_model(其transformer的参数)
# 输出：prompt embedding

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, wordset):
        super().__init__()
        self.n_cls = len(classnames)
        self.n_ctx = cfg.TRAINER.CAPG.N_CTX
        dtype = clip_model.dtype

        # 维度
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        vis_dim = clip_model.visual.output_dim  # 512

        # 图片分辨率
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # ***********************************************
        # prompt_prefix: "X X X X"
        prompt_prefix = " ".join(["X"] * self.n_ctx)
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        self.name_lens: list[int] = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, L)
        embedding = clip_model.token_embedding(self.tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS
        # ***********************************************

        # pg配置
        pg_config_path = os.path.join(os.getcwd(), cfg.TRAINER.CAPG.ROBERTA_PATH)
        self.pg_config = AutoConfig.from_pretrained(
            pg_config_path,
            hidden_size=ctx_dim,
            intermediate_size=ctx_dim * 4,
            num_labels=3,
            finetuning_task='QNLI'
        )
        # 根据超参数实例化pg
        self.pg = SPLayer(self.pg_config)
        # pg初始化
        reset_SPLayer_parameters(clip_model, self.pg)
        if cfg.TRAINER.CAPG.PREC == "fp16":
            self.pg.half()

        # Adapter
        self.adapter = Adapter(ctx_dim, self.n_ctx)
        if cfg.TRAINER.CAPG.PREC == "fp16":
            self.adapter.half()

        # wordset
        self.wordset = wordset
        self.wordset_tokenized = clip.tokenize(self.wordset)
        self.wordset_embedding = clip_model.token_embedding(self.wordset_tokenized).type(dtype)  # [1, L, ctx_dim]

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        # im_features [batch, ctx_dim]
        # image_embeddings
        im_embeddings = im_features.unsqueeze(1)  # [batch, 1, ctx_dim]

        # wordset_embeddings
        wordset_embeddings = self.wordset_embedding.expand(im_features.shape[0], -1, -1)  # [batch, L, ctx_dim]

        # generate prompt
        generated_prompts = self.adapter(self.pg(im_embeddings, wordset_embeddings))  # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = []
        for generated_prompts_i in generated_prompts:
            ctx_i = generated_prompts_i.unsqueeze(0).expand(self.n_cls, -1, -1)   # (n_cls, n_ctx, ctx_dim)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, L, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)  # [batch, n_cls, L, ctx_dim]

        return prompts  # [batch, n_cls, L, ctx_dim]


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, wordset):
        super().__init__()
        # 实例化prompt learner
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model, wordset)
        # (n_cls, L) text_encoder的初始输入
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def forward(self, image, label=None):
        # 温度参数
        logit_scale = self.logit_scale.exp()

        tokenized_prompts = self.tokenized_prompts  # (n_cls, L)

        # 图像特征
        image_features = self.image_encoder(image.type(self.dtype))  # (batch, C, H, W)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)   # (batch, n_cls, L, ctx_dim)

        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)

        # 损失函数L1
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits

# Trainer: 导入数据集，导入模型，训练及测试
@TRAINER_REGISTRY.register()
class CAPG(TrainerX):
    # 检查参数设置精度是否为["fp16", "fp32", "amp"]其中之一
    def check_cfg(self, cfg):
        assert cfg.TRAINER.CAPG.PREC in ["fp16", "fp32", "amp"]

    # 1.搭建模型
    def build_model(self):
        cfg = self.cfg
        # 类名
        classnames = self.dm.dataset.classnames

        # (1)加载CLIP模型到CPU中
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        # (2)设置clip_model精度为与cfg.TRAINER.CAPG.PREC保持一致
        if cfg.TRAINER.CAPG.PREC == "fp32" or cfg.TRAINER.CAPG.PREC == "amp":
            # CLIP's default precision is fp16
            # 即model.visual.conv1.weight.dtype
            clip_model.float()

        # (3) 构建模型：（输入参数：cfg, 类名， 预训练好的clip模型）
        # 再细看下 CustomCLIP，以及设置的可调节的参数, model.named_parameters():
        print("Building custom CLIP")

        # Prompt Words Set
        wordset = []
        templates = " ".join(IMAGENET_TEMPLATES).split(" ")
        for word in templates:
            if word not in wordset:
                wordset.append(word)
        wordset = " ".join(wordset)
        wordset.replace(' {}.', '')

        self.model = CustomCLIP(cfg, classnames, clip_model, wordset)

        # （4）除了model.named_parameters中prompt learner之外的参数,requires_grad_设置为false
        # 即设置可优化的参数仅为prompt learner的部分
        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        # prompt learner参数初始化
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.prompt_learner.wordset_embedding = self.model.prompt_learner.wordset_embedding.to(self.device)


        # 初始化后将prompt learner参数喂给optimizer
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.CAPG.PREC == "amp" else None

        # # 若有多个GPU，设置为并行训练
        # # Note that multi-gpu training could be slow because CLIP's size is
        # # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    # 前向传播和反向传播
    def forward_backward(self, batch):
        # 导入数据
        # image, image_n, label = self.parse_batch_train(batch)
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.CAPG.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward(retain_graph=True)
            optim.step()

        loss_summary = {"loss": loss.item()}

        # 训练一轮后，更新学习率
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    # 加载训练好的模型，用于测试
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
