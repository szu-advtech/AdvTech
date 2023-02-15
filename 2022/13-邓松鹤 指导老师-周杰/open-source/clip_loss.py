import torch
import clip
from clip_utils import background_dict, category_dict
import json


# maximize similarity
class SimMaxLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMaxLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(x + self.margin) * weights).mean()


# minimize similarity
class SimMinLoss(torch.nn.Module):

    def __init__(self, margin=0):
        super(SimMinLoss, self).__init__()
        self.margin = margin

    def forward(self, x, weights):
        x = x.clamp(0.0001, 0.9999)
        return -(torch.log(1 - x + self.margin) * weights).mean()


# suppress background activation
class BackgroundSuppressionLoss(torch.nn.Module):
    """
    based on threshold
    """

    def __init__(self, threshold=0.26, dname='coco'):
        super(BackgroundSuppressionLoss, self).__init__()
        self.dname = dname
        self.background = background_dict[dname]
        self.threshold = threshold
        print(f'Use CBSLoss! threshold: {threshold}')

    def forward(self, clip_model, images, eps=0.0001):
        image_features = clip_model.encode_image(images)  # [N1, C]
        text_features = clip_model.encode_text(clip.tokenize(self.background).cuda())  # [N2, C]

        # normalization
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = (image_features @ text_features.t())  # [N1, N2]
        mask = torch.zeros_like(logits_per_image)
        mask = torch.where(logits_per_image > self.threshold, torch.ones_like(mask), torch.zeros_like(mask))

        return -(torch.log(1 - logits_per_image) * mask).sum()


class BackgroundSuppressionLossWithLabel(torch.nn.Module):
    """
    based on threshold
    """

    def __init__(self, background_path=None, threshold=0.24, dname='coco'):
        super(BackgroundSuppressionLossWithLabel, self).__init__()
        self.dname = dname
        self.threshold = threshold
        self.use_mean = False
        if background_path is not None:
            self.backgrounds = json.load(open(background_path, 'r'))
        else:
            raise NotImplementedError
        print(
            f'Use CBSLossWithLabel! threshold: {threshold} use_mean: {self.use_mean} background_path: {background_path}')

    def forward(self, clip_model, images, labels, eps=0.0001):
        image_features = clip_model.encode_image(images)  # [N1, C]
        background_dict_keys = self.backgrounds.keys()
        loss = torch.tensor(.0, requires_grad=True, device=images.device)
        # scores = []
        for i in range(labels.size(0)):
            idx = torch.nonzero(labels[i], as_tuple=False).squeeze()
            category = category_dict['voc'][idx]
            if category not in background_dict_keys:
                print(f'L_CBS_label: no {category} in background_dict_keys')
                continue
            text = self.backgrounds[category]
            if len(text) == 0:
                continue
            text_features = clip_model.encode_text(clip.tokenize(text).cuda())  # [N2, C]

            image_feature = image_features[i].reshape(1, -1)  # reshape为一行

            # normalization
            image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.permute(1, 0)

            x = image_feature @ text_features
            if self.use_mean:
                loss = loss + (-(torch.log(1 - x)).mean())
                continue
            x = torch.where(x > self.threshold, x, torch.zeros_like(x))
            # scores.append(x.detach())

            loss = loss + -(torch.log(1 - x)).sum()  # 不能用 .item() loss无法反向传播
            # loss = loss + -(torch.log(1 - masked)).sum()
        # scores = torch.stack(scores)
        # loss = -(torch.log(1 - scores)).sum()
        return loss


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=1.0, bg_bg_pos=False, alpha=1.0, beta=1.0):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.use_bg_bg_pos = bg_bg_pos
        self.alpha = alpha
        self.beta = beta
        print(f'Use InfoNCELoss! temperature: {temperature} use_bg_bg_pos: {bg_bg_pos} hyper: [{alpha},{beta}]')

    def forward(self, clip_model, fg_imgs, fg_labels: list, bg_imgs, bg_labels: [list, ...]):
        fg_img_features = clip_model.encode_image(fg_imgs)  # [N1, C]
        bg_img_features = clip_model.encode_image(bg_imgs)  # [N1, C]

        positive_sims = torch.tensor(0., requires_grad=True, device=fg_imgs.device)
        negative_sims = torch.tensor(0., requires_grad=True, device=fg_imgs.device)
        for i in range(fg_imgs.size(0)):
            fg_label = fg_labels[i]
            bg_label = bg_labels[i]
            # delete fg_label from bg_label
            if fg_label in bg_label:
                bg_label.remove(fg_label)
            # prompt text -> 'a photo of {label}.'
            fg_label = f'a photo of {fg_label}.'
            bg_label = [f'a photo of {_}.' for _ in bg_label]
            fg_text_feature = clip_model.encode_text(clip.tokenize(fg_label).cuda())  # [1, C]
            bg_text_feature = clip_model.encode_text(clip.tokenize(bg_label).cuda())  # [N2, C]
            fg_text_feature = fg_text_feature / fg_text_feature.norm(dim=-1, keepdim=True)
            bg_text_feature = bg_text_feature / bg_text_feature.norm(dim=-1, keepdim=True)

            fg_img_feature = fg_img_features[i].reshape(1, -1)  # reshape为一行
            bg_img_feature = bg_img_features[i].reshape(1, -1)  # reshape为一行
            fg_img_feature = fg_img_feature / fg_img_feature.norm(dim=-1, keepdim=True)
            bg_img_feature = bg_img_feature / bg_img_feature.norm(dim=-1, keepdim=True)

            fg_img_bg_text_logits = fg_img_feature @ bg_text_feature.t()  # [1, N2]
            bg_img_fg_text_logits = bg_img_feature @ fg_text_feature.t()  # [1, 1]
            fg_img_fg_text_logits = fg_img_feature @ fg_text_feature.t()  # [1, 1]

            positive_sims = positive_sims + self.alpha * torch.exp(fg_img_fg_text_logits / self.temperature)
            negative_sims = negative_sims + \
                            torch.exp(bg_img_fg_text_logits / self.temperature).sum() + \
                            torch.exp(fg_img_bg_text_logits / self.temperature).sum()

            if self.use_bg_bg_pos:
                bg_img_bg_text_logits = bg_img_feature @ bg_text_feature.t()
                positive_sims = positive_sims + self.beta * torch.exp(bg_img_bg_text_logits / self.temperature).mean()

        loss = -torch.log(positive_sims / negative_sims)
        return loss
