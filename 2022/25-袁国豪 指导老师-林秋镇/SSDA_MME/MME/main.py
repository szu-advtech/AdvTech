import argparse
import os
import numpy as np
import torch
from torch import optim, nn
from torch.autograd import Variable
from data_loader import return_dataset
import data_loader, models
from loss import adentropy
from lr_schedule import inv_lr_scheduler
from utils import weights_init
import torch.nn.functional as F
seed=8
cuda=True

parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--dataset', type=str, default='office',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--source', type=str, default='webcam',
                    help='source domain')
parser.add_argument('--target', type=str, default='amazon',
                    help='target domain')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')

args = parser.parse_args()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

record_dir = 'record/%s/%s' % (args.dataset, 'MME')
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           ('MME', 'alexnet', args.source,
                            args.target, args.num))


source_loader, target_loader, target_unlabel_loader, target_validate_loader, \
    target_test_loader= return_dataset(args)



G = models.AlexNetBase()
F1 = models.Predictor()

params=[]

for key,value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params+=[{'params':[value],'lr':args.multi,'weight_decay':0.0005}]
        else:
            params+=[{'params':[value],'lr':args.multi*10,'weight_decay':0.0005}]

weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()


def train():
    G.train()
    F1.train()
    optimizer_g=optim.SGD(params, momentum=0.9,weight_decay=0.0005, nesterov=True)
    optimizer_f=optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,weight_decay=0.0005, nesterov=True)
    param_lr_g=[]
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f=[]
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion=nn.CrossEntropyLoss().cuda()
    all_step=args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_unlabel_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_unlabel_loader)
    best_acc = 0
    counter = 0
    train_loss = []
    train_Hloss = []
    train_tgtloss = []
    train_acc = []
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_unlabel_loader)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        try:
            source_data,source_label=data_iter_s.next()
        except Exception as err:
            data_iter_s = iter(source_loader)
            source_data,source_label=data_iter_s.next()
        try:
            target_data,target_label=data_iter_t.next()
        except Exception as err:
            data_iter_t = iter(target_loader)
            target_data,target_label=data_iter_t.next()
        try:
            target_unlabel_data,__=data_iter_t_unl.next()
        except Exception as err:
            data_iter_t_unl = iter(target_unlabel_loader)
            target_unlabel_data,__=data_iter_t_unl.next()
        if cuda:
            source_data,source_label=source_data.cuda(),source_label.cuda()
            target_data,target_label=target_data.cuda(),target_label.cuda()
            target_unlabel_data=target_unlabel_data.cuda()

        source_data,source_label=Variable(source_data),Variable(source_label)
        target_data,target_label=Variable(target_data),Variable(target_label)
        target_unlabel_data=Variable(target_unlabel_data)

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        data = torch.cat((source_data, target_data), 0)
        target = torch.cat((source_label, target_label), 0)
        output = G(data)
        out1 = F1(output)
        loss = criterion(out1, target.long())
        train_loss.append(loss.item())
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        output_tgt=G(target_unlabel_data)
        pseudolabels_target=F.softmax(output_tgt, 1)

        output = G(data)
        pseudolabels_source =F.softmax(output, 1)
        expectation_ratio = (1e-6 + torch.mean(pseudolabels_source)) / (1e-6 + torch.mean(pseudolabels_target))
        final_pseudolabels = F.normalize((pseudolabels_target * expectation_ratio), p=2, dim=1)

        loss_t=adentropy(F1, final_pseudolabels, args.lamda)
        train_Hloss.append(loss_t.item())
        loss_t.backward()
        optimizer_f.step()
        optimizer_g.step()
        log_train = 'S {} T {} Train Ep: {} lr{} \t ' 'Loss Classification: {:.6f} Loss T {:.6f} ' 'Method MME\n'.format(
            args.source, args.target,step, lr, loss.data,-loss_t.data)
        G.zero_grad()
        F1.zero_grad()
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 and step>0:
            loss_test, acc_test = test(target_test_loader)
            train_tgtloss.append(loss_test.item())
            loss_val, acc_val = test(target_validate_loader)
            G.train()
            F1.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break
            print('best acc test %f best acc val %f' % (best_acc_test,acc_val))
            train_acc.append(best_acc_test)
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %f final %f \n' % (step,
                                                         best_acc_test,
                                                         acc_val))
            G.train()
            F1.train()
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format('MME', args.source,
                                               args.target, step)))
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format('MME', args.source,
                                               args.target, step)))

    with open("./train_loss.txt", 'w') as train_los:
        train_los.write(str(train_loss))
    with open("./train_Hloss.txt", 'w') as train_Hlos:
        train_Hlos.write(str(train_Hloss))
    with open("./train_tgtloss.txt", 'w') as train_tgtlos:
        train_tgtlos.write(str(train_tgtloss))
    with open("./train_acc.txt", 'w') as train_ac:
        train_ac.write(str(train_acc))


def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = 31
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            data,target=data_t
            if cuda:
                data,target=data.cuda(),target.cuda()
            data,target=Variable(data),Variable(target)
            feat = G(data)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += data.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(target.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(target.data).cpu().sum()
            test_loss += criterion(output1, target.long()) / len(loader)
    print('\nTest set: Average loss: {:.4f}, ''Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


if __name__ == '__main__':

    train()