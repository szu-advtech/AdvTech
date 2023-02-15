import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from basics import task_loss, final_objective_loss, evaluate_steps
from utils.distributed import broadcast_coalesced, all_reduce_coalesced
from utils.io import save_results


def permute_list(list):
    indices = np.random.permutation(len(list))   # 按照给定列表生成一个打乱后的随机列表
    return [list[i] for i in indices]

# tensor就是n维数组，是神经网络的基本数据结构

class Trainer(object):
    def __init__(self, state, models):
        self.state = state
        self.models = models
        self.num_data_steps = state.distill_steps  # how much data we have          一个epoch需要的步数 default=10
        self.T = state.distill_steps * state.distill_epochs  # how many sc steps we run
        self.num_per_step = state.num_classes * state.distilled_images_per_class_per_step   # 每一步图片的数量 = 标签种类数量 * 每一步一个标签所需的图片数量
        assert state.distill_lr >= 0, 'distill_lr must >= 0'
        self.init_data_optim()

    def init_data_optim(self):
        self.params = []
        state = self.state
        optim_lr = state.lr

        # labels 设置生成一个epoch图片的标签（一个epoch包含num_data_steps个step）
        self.labels = []
        distill_label = torch.arange(state.num_classes, dtype=torch.long, device=state.device) \
                             .repeat(state.distilled_images_per_class_per_step, 1)  # [[0, 1, 2, ...], [0, 1, 2, ...]]
        distill_label = distill_label.t().reshape(-1)  # [0, 0, ..., 1, 1, ...]
        for _ in range(self.num_data_steps):
            self.labels.append(distill_label)
        self.all_labels = torch.cat(self.labels)

        # data 设置生成一个epoch的图片
        self.data = []
        for _ in range(self.num_data_steps):
            distill_data = torch.randn(self.num_per_step, state.nc, state.input_size, state.input_size,
                                       device=state.device, requires_grad=True)   #  返回一个符合均值为0，方差为1的正态分布（标准正态分布）中填充随机数的张量
            self.data.append(distill_data)
            self.params.append(distill_data)

        # lr

        # 设置生成图片的学习率
        # undo the softplus + threshold
        raw_init_distill_lr = torch.tensor(state.distill_lr, device=state.device)    # torch.tensor()是一个函数，是对张量数据的拷贝，根据传入data的类型来创建Tensor;
        # 生成T个step的初始学习率
        raw_init_distill_lr = raw_init_distill_lr.repeat(self.T, 1)
        self.raw_distill_lrs = raw_init_distill_lr.expm1_().log_().requires_grad_()
        self.params.append(self.raw_distill_lrs)

        assert len(self.params) > 0, "must have at least 1 parameter"

        # now all the params are in self.params, sync if using distributed
        if state.distributed:
            broadcast_coalesced(self.params)
            logging.info("parameters broadcast done!")

        # 优化器设置
        self.optimizer = optim.Adam(self.params, lr=state.lr, betas=(0.5, 0.999))
        # 调整学习率设置
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=state.decay_epochs,
                                                   gamma=state.decay_factor)
        for p in self.params:
            p.grad = torch.zeros_like(p)  # 产生一个维度和p一样大小的全0数组

    # 训练数据  得到一个step的合成数据
    def get_steps(self):
        # 得到一个generator，每次得到每一个epoch的一个step的数据
        data_label_iterable = (x for _ in range(self.state.distill_epochs) for x in zip(self.data, self.labels))
        # print(self.state.distill_epochs,len(self.data),len(self.labels))  # 3,10,10
        lrs = F.softplus(self.raw_distill_lrs).unbind()  # softplus--文档  lrs--学习率

        steps = []
        # 依次取一个epoch的一个step
        for (data, label), lr in zip(data_label_iterable, lrs):
            steps.append((data, label, lr))
        # print(steps[0][0].shape)  # torch.Size([10, 1, 28, 28])
        return steps


    # 第6、7行
    def forward(self, model, rdata, rlabel, steps):
        state = self.state
        # print(type(state))
        # print(state)

        # forward
        model.train()
        w = model.get_param()
        # print(w)
        params = [w]
        gws = []

        # Compute updated parameter with GD
        for step_i, (data, label, lr) in enumerate(steps):
            with torch.enable_grad():  # 启用渐变计算的上下文管理器。
                # 测试模型在w参数下在蒸馏数据下的效果
                output = model.forward_with_param(data, w)
                # print(state)
                # print(type(output))
                # print(output)
                # print(label)
                loss = task_loss(state, output, label)
            gw, = torch.autograd.grad(loss, w, lr.squeeze(), create_graph=True)  # 求loss关于w的导数

            with torch.no_grad():
                # new_x = w - gw 获得更新后的参数
                new_w = w.sub(gw).requires_grad_()
                params.append(new_w)
                gws.append(gw)
                w = new_w

        # final L
        # Evaluate the objective function on real training data
        model.eval()
        output = model.forward_with_param(rdata, params[-1])
        ll = final_objective_loss(state, output, rlabel)
        return ll, (ll, params, gws)  # 损失值数组 每次更新后的参数数组 每次更新的梯度数组

    # 第9行
    def backward(self, model, rdata, rlabel, steps, saved_for_backward):
        l, params, gws = saved_for_backward   # 损失值数组 每次更新后的参数数组 每次更新的梯度数组
        state = self.state

        datas = []
        gdatas = []
        lrs = []
        glrs = []

        dw, = torch.autograd.grad(l, (params[-1],))

        # backward
        model.train()
        # Notation:
        #   math:    \grad is \nabla
        #   symbol:  d* means the gradient of final L w.r.t. *
        #            dw is \d L / \dw
        #            dgw is \d L / \d (\grad_w_t L_t )
        # We fold lr as part of the input to the step-wise loss
        #
        #   gw_t     = \grad_w_t L_t       (1)
        #   w_{t+1}  = w_t - gw_t          (2)
        #
        # Invariants at beginning of each iteration:
        #   ws are BEFORE applying gradient descent in this step
        #   Gradients dw is w.r.t. the updated ws AFTER this step
        #      dw = \d L / d w_{t+1}
        for (data, label, lr), w, gw in reversed(list(zip(steps, params, gws))):
            # hvp_in are the tensors we need gradients w.r.t. final L:
            #   lr (if learning)
            #   data
            #   ws (PRE-GD) (needed for next step)
            #
            # source of gradients can be from:
            #   gw, the gradient in this step, whose gradients come from:
            #     the POST-GD updated ws
            hvp_in = [w]
            hvp_in.append(data)
            hvp_in.append(lr)
            dgw = dw.neg()  # gw is already weighted by lr, so simple negation
            hvp_grad = torch.autograd.grad(
                outputs=(gw,),
                inputs=hvp_in,
                grad_outputs=(dgw,)
            )
            # Update for next iteration, i.e., previous step
            with torch.no_grad():
                # Save the computed gdata and glrs
                datas.append(data)
                gdatas.append(hvp_grad[1])
                lrs.append(lr)
                glrs.append(hvp_grad[2])

                # Update for next iteration, i.e., previous step
                # Update dw
                # dw becomes the gradients w.r.t. the updated w for previous step
                dw.add_(hvp_grad[0])

        return datas, gdatas, lrs, glrs

    def accumulate_grad(self, grad_infos):
        bwd_out = []
        bwd_grad = []
        for datas, gdatas, lrs, glrs in grad_infos:
            bwd_out += list(lrs)
            bwd_grad += list(glrs)
            for d, g in zip(datas, gdatas):
                d.grad.add_(g)
        if len(bwd_out) > 0:
            torch.autograd.backward(bwd_out, bwd_grad)

    # 保存生成的图片和学习率
    def save_results(self, steps=None, visualize=True, subfolder=''):
        with torch.no_grad():
            steps = steps or self.get_steps()
            save_results(self.state, steps, visualize=visualize, subfolder=subfolder)

    def __call__(self):
        return self.train()

    def prefetch_train_loader_iter(self):
        state = self.state
        device = state.device
        # 生成一个train_loader迭代器
        train_iter = iter(state.train_loader)
        # 对每一个epoch (default: 400)
        for epoch in range(state.epochs):
            niter = len(train_iter)
            prefetch_it = max(0, niter - 2)
            for it, val in enumerate(train_iter):
                # Prefetch (start workers) at the end of epoch BEFORE yielding
                # 如果蒸馏数据集即将训练完一次，则再训练一次
                if it == prefetch_it and epoch < state.epochs - 1:
                    train_iter = iter(state.train_loader)
                yield epoch, it, val

    def train(self):
        state = self.state
        device = state.device
        train_loader = state.train_loader
        sample_n_nets = state.local_sample_n_nets
        grad_divisor = state.sample_n_nets  # i.e., global sample_n_nets
        ckpt_int = state.checkpoint_interval    # 检查点间隔

        data_t0 = time.time()
        # 需要蒸馏的数据集的每一个epoch
        for epoch, it, (rdata, rlabel) in self.prefetch_train_loader_iter():
            data_t = time.time() - data_t0

            # it=0表示蒸馏数据集已经训练一遍了
            if it == 0:
                # 调整学习率
                self.scheduler.step()

            # 如果蒸馏数据集训练一遍并且该epoch为检查点
            if it == 0 and ((ckpt_int >= 0 and epoch % ckpt_int == 0) or epoch == 0):
                with torch.no_grad():
                    steps = self.get_steps()  # 获得3个epoch的生成图片信息
                # 保存生成的图片和学习率
                self.save_results(steps=steps, subfolder='checkpoints/epoch{:04d}'.format(epoch))
                # 评估模型用蒸馏数据集训练前和训练后的效果
                evaluate_steps(state, steps, 'Begin of epoch {}'.format(epoch))

            do_log_this_iter = it == 0 or (state.log_interval >= 0 and it % state.log_interval == 0)

            self.optimizer.zero_grad()
            # 获得需要蒸馏的数据集
            rdata, rlabel = rdata.to(device, non_blocking=True), rlabel.to(device, non_blocking=True)

            if sample_n_nets == state.local_n_nets:
                tmodels = self.models
            else:
                idxs = np.random.choice(state.local_n_nets, sample_n_nets, replace=False)
                tmodels = [self.models[i] for i in idxs]

            t0 = time.time()
            losses = []
            steps = self.get_steps()

            # activate everything needed to run on this process
            grad_infos = []
            for model in tmodels:
                if state.train_nets_type == 'unknown_init':
                    model.reset(state)

                l, saved = self.forward(model, rdata, rlabel, steps)
                losses.append(l.detach())
                # 获得生成图片和学习率的更新梯度
                grad_infos.append(self.backward(model, rdata, rlabel, steps, saved))
                del l, saved
            # 更新生成图片和学习率
            self.accumulate_grad(grad_infos)

            # all reduce if needed
            # average grad
            all_reduce_tensors = [p.grad for p in self.params]
            # 如果记录这个iteration
            if do_log_this_iter:
                losses = torch.stack(losses, 0).sum()
                all_reduce_tensors.append(losses)

            # 如果是分布式
            if state.distributed:
                all_reduce_coalesced(all_reduce_tensors, grad_divisor)
            else:
                for t in all_reduce_tensors:
                    t.div_(grad_divisor)

            # opt step
            self.optimizer.step()
            t = time.time() - t0

            if do_log_this_iter:
                loss = losses.item()
                logging.info((
                    'Epoch: {:4d} [{:7d}/{:7d} ({:2.0f}%)]\tLoss: {:.4f}\t'
                    'Data Time: {:.2f}s\tTrain Time: {:.2f}s'
                ).format(
                    epoch, it * train_loader.batch_size, len(train_loader.dataset),
                    100. * it / len(train_loader), loss, data_t, t,
                ))
                if loss != loss:  # nan
                    raise RuntimeError('loss became NaN')

            del steps, grad_infos, losses, all_reduce_tensors

            data_t0 = time.time()

        with torch.no_grad():
            steps = self.get_steps()
        self.save_results(steps)
        return steps


def distill(state, models):
    return Trainer(state, models).train()
