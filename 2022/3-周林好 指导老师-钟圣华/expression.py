import torch
import torchvision
from torchvision import transforms

'''
表情分析模块
'''

# 待预测类别
classes = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']


# VGG16模型构建
def model_struct(num_cls):
    model_vgg16 = torchvision.models.vgg16(pretrained=True)  # 加载torch原本的vgg16模型，pretrained=True表示使用预训练模型
    num_fc = model_vgg16.classifier[6].in_features  # 获取最后一层的输入维度
    model_vgg16.classifier[6] = torch.nn.Linear(num_fc, num_cls)  # 修改最后一层的输出维度，即分类数
    # 对于模型的每个权重，使其不进行反向传播，即固定参数
    for param in model_vgg16.parameters():
        param.requires_grad = False
    # 将分类器的最后层输出维度换成了num_cls，这一层需要重新学习
    for param in model_vgg16.classifier[6].parameters():
        param.requires_grad = True
    model_vgg16.to('cpu')
    return model_vgg16


# 情绪预测
def predict(draw, model):
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(draw).to('cpu')
    img = torch.unsqueeze(img, dim=0)
    out = model(img)
    # print('out = ', out)      # out = tensor([[-38.2169, -71.5800, -35.2311, -34.2698, -34.9377, -34.3654, -31.7862]], grad_fn=<AddmmBackward0>)
    # print('out = ', out[0][0].item())     # out = -38.216888427734375

    pre = torch.max(out, 1)[1]
    cls = classes[pre.item()]

    # pre_num = torch.max(out).item()  # 预测具体数字     pre_num = -31.78618049621582
    # print(pre_num)

    # 情感得分
    score = out[0][6].item() + 2 * out[0][3].item() + 3 * out[0][5].item() - (
                out[0][2].item() + out[0][4].item() + 2 * out[0][1].item() + 2 * out[0][0].item())

    return cls, score  # 返回预测的情绪、情感得分


# 情感得分
def facial_expression(imageSave):
    # print(imageSave)
    # imageSave = imageSave.reshape((1, 224, 224, 3))

    device = torch.device('cpu')
    vgg_model = model_struct(7)
    vgg_model.to(device)
    vgg_model.eval()
    save = torch.load('model_data/model.pth', map_location=torch.device('cpu'))  # 调用权重
    vgg_model.load_state_dict(save['model'])
    exp, score = predict(imageSave, vgg_model)
    return exp, score  # 返回预测的情绪、情感得分
