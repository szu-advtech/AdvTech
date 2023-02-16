# import torch
# from Projector import Projector
#
# x = torch.zeros(2,2,2052,dtype=torch.float32)
#
# model = Projector()
#
# out = model(x)
#
#
#
#
#
#
#
#



# from transformers import GPT2Tokenizer, GPT2Model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# sents = ["I like\n",
#         "funk is a \n",
#         "shit man, this is"]
# # encoded_input = tokenizer(text, return_tensors='pt')
#
#
# tokenizer.add_special_tokens({'pad_token': '0'})
# #增强的编码函数
# out = tokenizer.batch_encode_plus(batch_text_or_text_pairs = [sents[0],
#         "funk is a",
#         "shit man, this is"],
#                                   padding = True,
#                                   truncation = True,
#                                   max_length = 10,
#                                   return_tensors = 'pt')
# fun = model.wte(out["input_ids"])
# print(out)
# print("asdasdasd")
# # print(tokenizer.decode(out['input_ids']))
#
#
# out.data.pop("input_ids")
# out.data["inputs_embeds"] = fun
# # output = model(**encoded_input)
# output = model(**out)
# print()

# cos测试
# import torch
#
# logit = torch.zeros((3,3))
# pp = torch.cosine_similarity(torch.randn(768),torch.randn(768),dim=0)
# logit[0,0] = pp
# print(pp)



# print(type(encoded_input))
# print(type(output))
# print(model)


# GPT测试2
# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# print(generator("The White man worked as a", max_length=10, num_return_sequences=5))
#
#
# set_seed(42)
# print(generator("The Black man worked as a", max_length=10, num_return_sequences=5))
#

#GPT测试3 从输入生成字


# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
#
# # 可选：如果您想了解发生的信息，请按以下步骤logger
# import logging
#
#
#
# # # 新增：
# # import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# # import torch.distributed as dist
# # from torch.nn.parallel import DistributedDataParallel as DDP
# # # 新增：从外面得到local_rank参数
# # import argparse
# # parser = argparse.ArgumentParser()
# # parser.add_argument("--local_rank", default=-1)
# # FLAGS = parser.parse_args()
# # local_rank = FLAGS.local_rank
# #
# # # 新增：DDP backend初始化
# # torch.cuda.set_device(local_rank)
# # dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
#
#
# logging.basicConfig(level=logging.INFO)
#
# # 加载预训练模型（权重）
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
#
# # 编码输入
# text = "Who was Jim Henson ? Jim Henson was a"
# indexed_tokens = tokenizer.encode(text)
#
# # 转换为PyTorch tensor
# tokens_tensor = torch.tensor([indexed_tokens])
# # 让我们看看如何使用GPT2LMHeadModel生成下一个跟在我们的文本后面的token：
#
# # 加载预训练模型（权重）
# model = GPT2LMHeadModel.from_pretrained('gpt2-large')
# model.to('cuda')
#
# # model = torch.nn.DataParallel(model)
# # # 新增：构造DDP model
# # model = DDP(model, device_ids=[local_rank], output_device=local_rank)
# # print(model)
# # 将模型设置为评估模式
# # 在评估期间有可再现的结果这是很重要的！
# # model.eval()
# model.train()
# # 如果你有GPU，把所有东西都放在cuda上
# tokens_tensor = tokens_tensor.to('cuda')
#
#
# #########################################################################################
# funkkk = "You are a navigation robot." \
#                 " You need to go to the destination according to the instructions and then stop." \
#                 " Your instruction is : go to the bedroom." \
#          "The actions you can choose are: A.;B.;C;"\
# "Choose a letter to indicate the direction of progress. The letter you choose is : "
# sents = [funkkk,
#         "funk is a \n",
#         "shit man, this is"]
# # encoded_input = tokenizer(text, return_tensors='pt')
#
#
# tokenizer.add_special_tokens({'pad_token': '0'})
# #增强的编码函数
# out = tokenizer.batch_encode_plus(batch_text_or_text_pairs = [sents[0],
#         "funk is a\n\n",
#         "shit man, this is"],
#                                   padding = True,
#                                   truncation = True,
#                                   max_length = 10,
#                                   return_tensors = 'pt').to('cuda')
# fun = model.transformer.wte(out["input_ids"]).to('cuda')
# print(out)
# print("asdasdasd")
# # print(tokenizer.decode(out['input_ids']))
#
# abc_index = " a b c d e f g h i j k l m n o p q r s t u v w x y z . ;"
# abc_tok = tokenizer(abc_index, return_tensors='pt')["input_ids"].cuda()
# abcc = tokenizer.decode(abc_tok[0])
# abcc2 = tokenizer.decode(abc_tok[0, 0])
# abcc3 = tokenizer.decode(abc_tok[0, 1])
# abcc4 = tokenizer.decode(abc_tok[0, 2])
# abcc5 = tokenizer.decode(abc_tok[0, 3])
# abcc6 = tokenizer.decode(abc_tok[0, 4])
# abcc7 = tokenizer.decode(abc_tok[0, 5])
# abcc8 = tokenizer.decode(abc_tok[0, 6])
# abcc9 = tokenizer.decode(abc_tok[0, 7])
# abcc10 = tokenizer.decode(abc_tok[0, 8])
# abcc11 = tokenizer.decode(abc_tok[0, 9])
# abcc12 = tokenizer.decode(abc_tok[0, 10])
# abcc13 = tokenizer.decode(abc_tok[0, 11])
# abcc14 = tokenizer.decode(abc_tok[0, 12])
# abcc15 = tokenizer.decode(abc_tok[0, 27])
# abcc16 = tokenizer.decode(abc_tok[0, 26])
# abcc17 = tokenizer.decode(abc_tok[0, 15])
#
#
# out.data.pop("input_ids")
# out.data["inputs_embeds"] = fun
# # output = model(**encoded_input)
# #########################################################################################
#
# # 预测所有标记
# with torch.no_grad():
#     # outputs = model(tokens_tensor)  # [1, 11, 50257],[2, 1, 12, 11, 64]
#     outputs = model(**out)
#
#     predictions = outputs[0]
# # 得到预测的下一个子词（在我们的例子中，是“man”这个词）
# predicted_index = torch.argmax(predictions[0, -1, :]).item()
# predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
# print()


# assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'
# 每个模型架构（Bert、GPT、GPT-2、Transformer XL、XLNet和XLM）的每个模型类的示例，可以在文档中找到。

# 使用过去的GPT-2
# 以及其他一些模型（GPT、XLNet、Transfo XL、CTRL），使用past或mems属性，这些属性可用于防止在使用顺序解码时重新计算键/值对。它在生成序列时很有用，因为注意力机制的很大一部分得益于以前的计算。
#
# 下面是一个使用带past的GPT2LMHeadModel和argmax解码的完整工作示例（只能作为示例，因为argmax decoding引入了大量重复）：

# 加载预训练模型（权重）
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 可选：如果您想了解发生的信息，请按以下步骤logger
import logging
logging.basicConfig(level=logging.INFO)

# 加载预训练模型（权重）
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

# 编码输入
text = "Who was Jim Henson ? Jim Henson was a"
indexed_tokens = tokenizer.encode(text)

# 转换为PyTorch tensor
tokens_tensor = torch.tensor([indexed_tokens])

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')

# 将模型设置为评估模式
# 在评估期间有可再现的结果这是很重要的！
model.eval()

# 如果你有GPU，把所有东西都放在cuda上
tokens_tensor = tokens_tensor.to('cuda')
# model.to('cuda')


generated = tokenizer.encode("The Manhattan bridge")
context = torch.tensor([generated])
past = None
input = tokenizer.batch_encode_plus(batch_text_or_text_pairs=["The Manhattan bridge"],
                                         # padding=True,
                                         truncation=True,
                                         max_length=100,
                                         return_tensors='pt',
                                    past_key_values = None)
# # 预测所有标记
# with torch.no_grad():
#     outputs = model(tokens_tensor)
#     predictions = outputs[0]
#
# # 得到预测的下一个子词（在我们的例子中，是“man”这个词）
#
# predicted_index = torch.argmax(predictions[0, -1, :]).item()

tasks_emb = model.transformer.wte(input["input_ids"])

input.data.pop("attention_mask")
for i in range(10):
    print(i)
    output = model(**input)
    token = torch.argmax(output[0][0, -1, :])



    generated += [token.tolist()]

    input.data.pop("input_ids")

    input.data["input_ids"] = token.unsqueeze(0).unsqueeze(0)
    input.data["past_key_values"] = output["past_key_values"]


sequence = tokenizer.decode(generated)


print(sequence)

# predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])




# assert predicted_text == 'Who was Jim Henson? Jim Henson was a man'

