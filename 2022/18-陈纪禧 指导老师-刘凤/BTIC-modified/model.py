import torch
import torch.nn as nn
from dataset import image_path
from PIL import Image
import numpy as np
from transformers import BertTokenizer, BertModel, ConvNextModel, ConvNextFeatureExtractor,\
                        AutoFeatureExtractor, SwinModel

BERT_PATH = './model/bert-base'
CONV_PATH = './model/convnext-tiny'
SWIN_PATH = './model/swin-tiny'


class BTIC(nn.Module):
    def __init__(self, text, image, device):
        super(BTIC, self).__init__()
        self.text = text
        self.image = image
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.bertmodel_t = BertModel.from_pretrained(BERT_PATH)
        self.transformer1 = nn.Transformer(d_model=768, nhead=8,
                                           num_encoder_layers=6,
                                           num_decoder_layers=6,
                                           dim_feedforward=516)
        self.attention = nn.MultiheadAttention(768, 8)
        self.fci1 = nn.Linear(2048, 1024)
        self.fci2 = nn.Linear(1024, 768)
        self.pool1 = nn.MaxPool2d(kernel_size=(32, 32))
        self.pool2 = nn.AvgPool2d(kernel_size=(10, 1))
        self.fc = nn.Linear(24, 2)
        self.dropout = nn.Dropout(0.5)

    def text_read(self, Id):
        """ 输入 batch id，读取出对应的文字，并使用 tokenizer 处理 """
        indices = Id.cpu().numpy().squeeze()
        # print(f"debug | indices: {indices}, shape: {indices.shape}")
        text_df_indices = self.text.iloc[indices]
        tx = text_df_indices["title"].values.tolist()
        tokens_pt = self.tokenizer(
            tx,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_attention_mask=False,
            return_tensors='pt',
        ).to(self.device)
        outputs = self.bertmodel_t(**tokens_pt)
        last_hidden_state = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output.unsqueeze(1)

        return last_hidden_state  # pooler_output

    def img_read(self, Id):
        indices = Id.cpu().numpy().squeeze()
        # 注意 image 是 ndarray 不是 df
        img_feature_indices = self.image[indices, 1]
        # img_feature_indices = np.array(img_feature_indices, dtype=float)
        # print(f"features: {img_feature_indices}, "
        #       f"shape: {img_feature_indices.shape}, dtype: {img_feature_indices.dtype}")
        img = torch.tensor(img_feature_indices.tolist()).to(self.device)
        img = img.squeeze(1)
        return img

    def textimage(self, Id):
        text = self.text_read(Id)
        text = text.permute(1, 0, 2)
        img = self.img_read(Id)
        # 768-dim 的话就不做 fc 了
        img = self.fci1(img)
        img = self.fci2(img)
        img = img.permute(1, 0, 2)
        img = self.transformer1(img, img)
        x = torch.cat((text, img), 0)

        outputx, weights = self.attention(x, x, x)
        x = outputx.permute(1, 0, 2).unsqueeze(1)
        x = self.pool1(x)
        x = self.pool2(x).squeeze(2).squeeze(1)
        return x

    def forward(self, Id):
        x = self.textimage(Id)
        x0 = self.dropout(x)
        x0 = self.fc(x0)
        x0 = x0.squeeze(1)
        logit = torch.sigmoid(x0)

        return logit, x


# convnext 和 swin 的作用是一样的
class BTICSwin(nn.Module):
    def __init__(self, text, device):
        super(BTICSwin, self).__init__()
        self.text = text
        self.device = device

        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
        self.bertmodel_t = BertModel.from_pretrained(BERT_PATH)

        # self.img_extractor = ConvNextFeatureExtractor.from_pretrained(CONV_PATH)
        # self.img_model = ConvNextModel.from_pretrained(CONV_PATH)

        self.img_extractor = AutoFeatureExtractor.from_pretrained(SWIN_PATH)
        self.img_model = SwinModel.from_pretrained(SWIN_PATH)

        self.transformer = nn.Transformer(d_model=768, nhead=8,
                                          num_encoder_layers=6,
                                          num_decoder_layers=6,
                                          dim_feedforward=516)
        self.attention = nn.MultiheadAttention(768, 8)
        self.pool1 = nn.MaxPool2d(kernel_size=(32, 32))
        self.pool2 = nn.AvgPool2d(kernel_size=(9, 1))
        self.fc = nn.Linear(24, 2)
        self.dropout = nn.Dropout(0.5)

    def text_read(self, Id):
        """ 输入 batch id，读取出对应的文字，并使用 tokenizer 处理 """
        indices = Id.cpu().numpy().squeeze()
        # print(f"debug | indices: {indices}, shape: {indices.shape}")
        text_df_indices = self.text.iloc[indices]
        tx = text_df_indices["title"].values.tolist()
        tokens_pt = self.tokenizer(
            tx,
            add_special_tokens=True,
            max_length=256,
            truncation=True,
            padding='max_length',
            return_attention_mask=False,
            return_tensors='pt',
        ).to(self.device)
        outputs = self.bertmodel_t(**tokens_pt)
        last_hidden_state = outputs.last_hidden_state
        # pooler_output = outputs.pooler_output.unsqueeze(1)

        return last_hidden_state

    def img_read(self, Id):
        # 这里的 Id 是 all data 表里的 index
        indices = Id.cpu().numpy().squeeze()
        text_df_indices = self.text.iloc[indices]
        image_ids = text_df_indices['id']
        image_list = list(map(lambda image_id: f"{image_path}/{image_id}.jpg", image_ids))
        images = list(map(lambda image: Image.open(image).convert("RGB"), image_list))

        images_pt = self.img_extractor(images, return_tensors="pt").to(self.device)
        outputs = self.img_model(**images_pt)
        last_hidden_state = outputs.last_hidden_state

        # 对于 convnext 而言，[batch, 768, 7, 7] 要转成 [batch, 49, 768]
        if last_hidden_state.shape[1:] != torch.Size([49, 768]):
            # (batch, 768, 7, 7) -> (batch, 768, 49)
            last_hidden_state = torch.flatten(last_hidden_state, start_dim=2)
            # (batch, 768, 49) -> (batch, 49, 768)
            last_hidden_state = torch.permute(last_hidden_state, dims=(0, 2, 1))

        assert last_hidden_state.shape[1:] == torch.Size([49, 768])
        return last_hidden_state

    def textimage(self, Id):
        text = self.text_read(Id)
        text = text.permute(1, 0, 2)
        img = self.img_read(Id)
        img = img.permute(1, 0, 2)
        img = self.transformer(img, img)
        x = torch.cat((text, img), dim=0)

        outputx, weights = self.attention(x, x, x)
        x = outputx.permute(1, 0, 2).unsqueeze(1)
        x = self.pool1(x)
        x = self.pool2(x).squeeze(2).squeeze(1)
        return x

    def forward(self, Id):
        x = self.textimage(Id)
        x0 = self.dropout(x)
        x0 = self.fc(x0)
        x0 = x0.squeeze(1)
        logit = torch.sigmoid(x0)

        return logit, x
