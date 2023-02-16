import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,5,6"
import torch
from torch import nn
import torch.autograd as autograd
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll

from torch.utils.data import DataLoader
from score.conditional_model import ConditionalModel
from score.ema import EMA
from score.process import p_sample_loop, noise_estimation_loss
from score.loss import loss_likelihood_bound



def sample_batch(size, noise=0.5):
    x, _ = make_swiss_roll(size, noise=noise)
    return x[:, [0, 2]] / 10.0


# 测试数据集是否能够显示
# def show_swiss_roll():
data = sample_batch(10 ** 4).T
device = "cuda" if torch.cuda.is_available() else "cpu"

# plt.figure(figsize=(16, 12))
# plt.scatter(*data, alpha=0.5, color='blue', edgecolors='white', s=60)
# plt.show()


def sliced_score_matching(model, samples):
    samples.requires_grad_(True)
    # 构建随机变量
    vectors = torch.randn_like(samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)
    # 计算优化后的jacobian矩阵
    logp, jvp = autograd.functional.jvp(model, samples, vectors, create_graph=True)
    # 计算范数损失
    norm_loss = (logp * vectors) ** 2 / 2.
    # 计算雅可比矩阵损失
    v_jvp = jvp * vectors
    jacob_loss = v_jvp
    loss = jacob_loss + norm_loss
    return loss.mean(-1).mean(-1)


# 实现去噪分数匹配损失
def denising_score_matching(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    return loss


# def net():
#     # 近似模型
#     model = nn.Sequential(
#         nn.Linear(2, 128), nn.Softplus(),
#         nn.Linear(128, 128), nn.Softplus(),
#         nn.Linear(128, 2)
#     )
#     # 创建ADAM优化器
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     dataset = torch.tensor(data.T).float()
#     for t in range(5000):
#         # 估计损失
#         loss = denising_score_matching(model, dataset)
#         # 在向后传递之前，将所有网络梯度归零
#         optimizer.zero_grad()
#         # 反向传递:计算损失相对于参数的梯度
#         loss.backward()
#         # 调用step函数来更新参数
#         optimizer.step()
#
#         if ((t % 1000) == 0):
#             print(loss)


# 我们可以观察到，我们的模型已经学会了通过在输入空间上绘制输出值来表示

def plot_gradients(model, data, plot_scatter=True):
    xx = np.stack(np.meshgrid(np.linspace(-1.5, 2.0, 50), np.linspace(-1.5, 2.0, 50)), axis=-1).reshape(-1, 2)
    scores = model(torch.from_numpy(xx).float()).detach()
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    plt.figure(figsize=(16, 12))
    if (plot_scatter):
        plt.scatter(*data, alpha=0.3, color='red', edgecolor='white', s=40)
    plt.quiver(xx.T[0], xx.T[1], scores_log1p[:, 0], scores_log1p[:, 1], width=0.002, color='black')
    plt.xlim(-1.5, 2.0)
    plt.ylim(-1.5, 2.0)
# n_steps=100
# model = ConditionalModel(n_steps)
# plot_gradients(model, data)
# plt.show()

# 朗之万抽样
def sample_langevin(model, x, n_steps=10, eps=1e-3, decay=.9, temperature=1.0):
    x_sequence = [x.unsqueeze(0)]
    for s in range(n_steps):
        z_t = torch.rand(x.size())
        x = x + (eps / 2) * model(x) + (np.sqrt(eps) * temperature * z_t)
        x_sequence.append(x.unsqueeze(0))
        eps *= decay
    return torch.cat(x_sequence)


# x = torch.Tensor([1.5, -1.5])
# samples = sample_langevin(model, x).detach()
# plot_gradients(model, data)
# plt.scatter(samples[:, 0], samples[:, 1], color='green', edgecolor='white', s=150)
#
# deltas = (samples[1:] - samples[:-1])
# deltas = deltas - deltas / torch.tensor(np.linalg.norm(deltas, keepdims=True, axis=-1)) * 0.04
# for i, arrow in enumerate(deltas):
#     plt.arrow(samples[i,0], samples[i,1], arrow[0], arrow[1], width=1e-4, head_width=2e-2, color="green", linewidth=3)

# plt.show()

# fig, axs = plt.subplots(1, 10, figsize=(28, 3))
# for i in range(10):
#     q_i = q_sample(dataset, torch.tensor([i * 10]))
#     axs[i].scatter(q_i[:, 0], q_i[:, 1], s=10)
#     axs[i].set_axis_off()
#     axs[i].set_title('$q(\mathbf{x}_{' + str(i * 10) + '})$')
# plt.show()

# @torch.no_grad()
# def sample_plot_image():
#     # 样本噪声
#     img_size = 64
#     img = torch.randn((1, 3, img_size, img_size), device=device)
#     plt.figure(figsize=(15, 15))
#     plt.axis('off')
#     num_images = 1
#     stepsize = int(T / num_images)
#
#     for i in range(0, T)[::-1]:
#         t = torch.full((1,), i, device=device, dtype=torch.long)
#         img = sample_timestep(img, t)
#         if i % stepsize == 0:
#             plt.subplot(1, num_images, int(i / stepsize + 1))
#             show_tensor_image(img.detach().cpu())
#     plt.show()



# Training



n_steps = 100

model = ConditionalModel(n_steps)
if torch.cuda.device_count() > 1:
    print(123)
    model = nn.DataParallel(model)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
dataset = torch.tensor(data.T).float()
# Create EMA model
ema = EMA(0.9)
ema.register(model)
# Batch size
BATCH_SIZE = 64
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
epochs = 100
for epoch in range(epochs):
    for step,batch in enumerate(dataloader):
        optimizer.zero_grad()
        loss = noise_estimation_loss(model, batch[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        ema.update(model)
        if step == 0:
            print(f" Epoch {epoch}  |  step {step:03d}  Loss:{loss.item()}")
            # image = batch[0]
            # show_tensor_image(image)
            print(loss)
            x_seq = p_sample_loop(model, dataset.shape)
            fig, axs = plt.subplots(1, 10, figsize=(28, 3))
            for i in range(1, 11):
                cur_x = x_seq[i * 10].detach()
                axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1], s=10)
                #axs[i-1].set_axis_off();
                axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*100)+'})$')
            plt.show()
# for t in range(1000):
#     # X is a torch Variable
#     permutation = torch.randperm(dataset.size()[0])
#     for i in range(0, dataset.size()[0], batch_size):
#         # Retrieve current batch
#         indices = permutation[i:i+batch_size]
#         batch_x = dataset[indices]
#         # Compute the loss.
#         loss = noise_estimation_loss(model, batch_x)
#         # Before the backward pass, zero all of the network gradients
#         optimizer.zero_grad()
#         # Backward pass: compute gradient of the loss with respect to parameters
#         loss.backward()
#         # Perform gradient clipping
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
#         # Calling the step function to update the parameters
#         optimizer.step()
#         # Update the exponential moving average
#         ema.update(model)
#     # Print loss
#     if (t % 100 == 0):
#         print(loss)
#         x_seq = p_sample_loop(model, dataset.shape)
#         fig, axs = plt.subplots(1, 10, figsize=(28, 3))
#         for i in range(1, 11):
#             cur_x = x_seq[i * 10].detach()
#             axs[i-1].scatter(cur_x[:, 0], cur_x[:, 1], s=10);
#             #axs[i-1].set_axis_off();
#             axs[i-1].set_title('$q(\mathbf{x}_{'+str(i*100)+'})$')
