import time
import model
import utils
from utils import args, tqdm
import torch
import numpy as np

class MFLoss(torch.nn.Module):
    def __init__(self):
        super(MFLoss, self).__init__()

    def forward(self, y, label):
        loss = -(label*torch.log(y) + (1-label)*torch.log(1-y)).sum(dim=1)
        return loss

class Train:
    def __init__(self, data, hidden_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.graph = data.graph
        self.has_train = False
        self.hidden_size = hidden_size
        self.node_num = len(data.vid2node)
        self.model = model.MGNN(self.graph, self.hidden_size, self.node_num)
        self.model.to(self.device)


    def train(self):
        brk = 0
        best_vali = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2_all)
        # LossFunc = MFLoss()
        LossFunc = torch.nn.BCEWithLogitsLoss()
        LossFunc.to(self.device)
        data_generator = self.data.gen_train_batch_for_train(args.batch_size)
        for ep in range(args.epochs):
            # pbar = tqdm(total=int(args.nb_train/args.batch_size), desc='training', leave=False)
            pbar = tqdm(total=args.nb_vali_step, desc='training', leave=False)
            loss = []
            t0 = time.time()
            self.model.update_h1h2()
            for _ in range(args.nb_vali_step):
                # [0]=click [1]=buy [2]=target [3]=negative
                data = next(data_generator)

                buy_list = torch.tensor(data[1]).long().to(self.device)
                click_list = torch.tensor(data[0]).long().to(self.device)
                target = torch.tensor(data[2]).long().to(self.device)
                target = target.reshape(target.shape[0], 1)
                neg = torch.tensor(data[3]).long().to(self.device)
                y, label = self.model(buy_list, click_list, target, neg)
                optimizer.zero_grad()
                _loss = LossFunc(y, label.float())
                _loss.backward(retain_graph=True)
                optimizer.step()
                loss.append(_loss)
                pbar.update(1)
            pbar.close()
            train_time = time.time() - t0

            vali_v, vali_str = self.metric('vali')
            if vali_v > best_vali:
                brk = 0
                best_vali = vali_v
                torch.save(self.model, f'./{utils.save_dir}/{args.run_name}-model.ckpt')
                # self.model.save()
            else:
                brk += 1
            red = (brk == 0)

            msg = f'#{ep + 1}/{args.epochs} loss: {torch.tensor(loss).mean():.5f}, brk: {brk}, vali: {vali_str}'
            if args.show_test and args.nb_test > 0:
                _, test_str = self.metric('test')
                msg = f'{msg}, test: {test_str}'
            vali_time = time.time() - t0 - train_time
            msg = f'{msg}, time: {train_time:.0f}s,{vali_time:.0f}s'

            args.log.log(msg, red=red)

            if ep < 60:
                brk = 0
            if brk >= args.early_stopping:
                break
            self.has_train = True

    def final_test(self):
        self.model = torch.load(f'./{utils.save_dir}/{args.run_name}-model.ckpt')
        _, ret = self.metric('test')
        return ret


    def metric(self, name):
        data_gen = self.data.gen_metric_batch(name, batch_size=256)
        pred_list, true_vid = self.topk(data_gen)

        pred_list = np.array(pred_list)
        true_vid = np.expand_dims(np.array(true_vid), -1)

        k = 20
        acc_ar = (pred_list == true_vid)[:, :k]  # [BS, K]
        acc = acc_ar.sum(-1)

        rank = np.argmax(acc_ar[:, :k], -1) + 1
        mrr = (acc / rank).mean()
        ndcg = (acc / np.log2(rank + 1)).mean()

        acc = acc.mean()
        # print(acc_ar)
        # print(mrr)
        # input()
        acc *= 100
        mrr *= 100
        ndcg *= 100
        ret = acc
        # ret = acc + mrr * 10 + ndcg * 5
        # ret = acc + mrr + ndcg
        # return ret, 'HR%:{:.3f},MRR%:{:.4f},NDCG%:{:.4f}'.format(acc, mrr, ndcg)
        return ret, '{:.3f},{:.4f},{:.4f}'.format(acc, mrr, ndcg)

    def topk(self, data_gen):
        pred_list = []
        true_vid = []
        cnt = 0
        pbar = tqdm(desc='predicting...', leave=False)
        for data in data_gen:
            buy_list = torch.tensor(data[1]).long().to(self.device)
            click_list = torch.tensor(data[0]).long().to(self.device)
            v, i = self.model.topk(buy_list,click_list)
            pred_list.extend(i.tolist())
            true_vid.extend(data[2])
            pbar.update(1)
            cnt += 1
            if args.run_test and cnt > 3:
                break
        pbar.close()
        return pred_list, true_vid


def main():
    print('hello world, Train.py')


if __name__ == '__main__':
    main()
