# coding=utf-8

import torch
from transformers.models.bert import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-base-cold')
model = BertForSequenceClassification.from_pretrained('thu-coai/roberta-base-cold')
model.eval()

# texts = ['你就是个傻逼！','长得像猪八戒猪八戒,特别喜欢呆在家里吹自己喜欢的牛,跟谁都不说话,拉车,帮人家拖地,后面还有个领着小朋友拉屎。','男女平等，黑人也很优秀。']
#
# model_input = tokenizer(texts,return_tensors="pt",padding=True)
# model_output = model(**model_input, return_dict=False)
# prediction = torch.argmax(model_output[0].cpu(), dim=-1)
# prediction = [p.item() for p in prediction]
# print(prediction)

f = open('test_cpm.txt','r',encoding='utf-8').readlines()
sum = 0
texts = ['1']
for text in f.readlines():
    post = text.find('title:')
    if post != -1:
        continue
    text = text.strip().replace('\n','')[22:142]
    texts[0] = text
    model_input = tokenizer(texts,return_tensors="pt",padding=True)
    model_output = model(**model_input, return_dict=False)
    prediction = torch.argmax(model_output[0].cpu(), dim=-1)
    prediction = [p.item() for p in prediction]
    sum += prediction[0]
    print(prediction)
print(sum)
# sum=0
# a= open('offensive_cpm.txt','r',encoding='utf-8')
# for text in a:
#     text.strip()
#     print(text)
#     model_input = tokenizer(texts, return_tensors="pt", padding=True)
#     model_output = model(**model_input, return_dict=False)
#     prediction = torch.argmax(model_output[0].cpu(), dim=-1)
#     prediction = [p.item() for p in prediction]
#     print(prediction)  # --> [1, 1, 0] (0 for Non-Offensive, 1 for Offenisve)
#     sum+=prediction[0]
# print(sum)

