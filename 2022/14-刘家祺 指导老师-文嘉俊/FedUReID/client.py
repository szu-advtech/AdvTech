import time
import torch
from utils import get_optimizer, get_model
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from optimization import Optimization
from client_loss import ExLoss
import warnings
class Client():
    def __init__(self, cid, data, device, project_dir, model_name, local_epoch, lr, batch_size, drop_rate, stride,
                 merge_percent, size_penalty):
        self.cid = cid
        self.project_dir = project_dir
        self.model_name = model_name
        self.data = data
        self.device = device
        self.local_epoch = local_epoch
        self.lr = lr
        self.batch_size = batch_size
        
        self.dataset_sizes = self.data.train_dataset_sizes[cid]
        self.train_loader = self.data.train_loaders[cid]

        self.full_model = get_model(self.data.train_class_sizes[cid], drop_rate, stride)
        # self.classifier = self.full_model.classifier.classifier
        # self.full_model.classifier.classifier = nn.Sequential()
        self.model = self.full_model
        self.distance=0
        self.optimization = Optimization(self.train_loader, self.device)
        # print("class name size",class_names_size[cid])
        self.nums_to_merge = int(self.dataset_sizes * merge_percent)
        self.size_penalty = size_penalty
        self.labels = []
        self.data = []

    def train(self, federated_model, use_cuda):
        self.y_err = []
        self.y_loss = []

        self.model.load_state_dict(federated_model.state_dict())
        # self.model.classifier.classifier = self.classifier
        # self.old_classifier = copy.deepcopy(self.classifier)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        # print(self.dataset_sizes)
        criterion = ExLoss(num_features = 512, num_classes = self.dataset_sizes, t=10).cuda()

        since = time.time()

        print('Client', self.cid, 'start training')
        for epoch in range(self.local_epoch):
            print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
            print('-' * 10)

            scheduler.step()
            self.model.train(True)
            running_loss = 0.0
            running_corrects = 0.0
            
            for data in self.train_loader:
                inputs, labels = data
                b, c, h, w = inputs.shape
                if b < self.batch_size:
                    continue
                if use_cuda:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                
                optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # warnings.filterwarnings("ignore")
                loss, _ = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item() * b
                running_corrects += float(torch.sum(preds == labels.data))

            used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
            epoch_loss = running_loss / used_data_sizes
            epoch_acc = running_corrects / used_data_sizes

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))

            self.y_loss.append(epoch_loss)
            self.y_err.append(1.0-epoch_acc)

            time_elapsed = time.time() - since
            print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Client', self.cid, 'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)
        
        # self.classifier = self.model.classifier.classifier
        # self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
        # self.model.classifier.classifier = nn.Sequential()

    def generate_soft_label(self, x, regularization):
        return self.optimization.kd_generate_soft_label(self.model, x, regularization)

    def get_model(self):
        return self.model

    def get_data_sizes(self):
        return self.dataset_sizes

    def get_train_loss(self):
        return self.y_loss[-1]

    def get_cos_distance_weight(self):
        return self.distance

    def update_data(self):
        labels = self.labels

        u_feas, feature_avg, label_to_images, fc_avg = self.generate_average_feature(labels)

        dists = self.calculate_distance(u_feas)

        idx1, idx2 = self.select_merge_data(u_feas, labels, label_to_images, self.size_penalty, dists)

        new_train_data, labels = self.generate_new_train_data(idx1, idx2, labels, self.nums_to_merge)

        num_train_ids = len(np.unique(np.array(labels)))

        # change the criterion classifer
        self.criterion = ExLoss(self.embeding_fea_size, num_train_ids, t=10).cuda()
        # new_classifier = fc_avg.astype(np.float32)
        # self.criterion.V = torch.from_numpy(new_classifier).cuda()

        self.labels, self.data = labels, new_train_data

    def calculate_distance(self, u_feas):
        # calculate distance between features
        x = torch.from_numpy(u_feas)
        y = x
        m = len(u_feas)
        dists = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(y, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        dists.addmm_(1, -2, x, y.t())
        return dists

    def select_merge_data(self, u_feas, label, label_to_images, ratio_n, dists):
        dists.add_(torch.tril(100000 * torch.ones(len(u_feas), len(u_feas))))

        cnt = torch.FloatTensor([len(label_to_images[label[idx]]) for idx in range(len(u_feas))])
        dists += ratio_n * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))

        for idx in range(len(u_feas)):
            for j in range(idx + 1, len(u_feas)):
                if label[idx] == label[j]:
                    dists[idx, j] = 100000

        dists = dists.numpy()
        ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
        idx1 = ind[0]
        idx2 = ind[1]
        return idx1, idx2

    def generate_new_train_data(self, idx1, idx2, label, num_to_merge):
        correct = 0
        num_before_merge = len(np.unique(np.array(label)))
        # merge clusters with minimum dissimilarity
        for i in range(len(idx1)):
            label1 = label[idx1[i]]
            label2 = label[idx2[i]]
            if label1 < label2:
                label = [label1 if x == label2 else x for x in label]
            else:
                label = [label2 if x == label1 else x for x in label]
            if self.u_label[idx1[i]] == self.u_label[idx2[i]]:
                correct += 1
            num_merged = num_before_merge - len(np.sort(np.unique(np.array(label))))
            if num_merged == num_to_merge:
                break

        # set new label to the new training data
        unique_label = np.sort(np.unique(np.array(label)))
        for i in range(len(unique_label)):
            label_now = unique_label[i]
            label = [i if x == label_now else x for x in label]
        new_train_data = []
        for idx, data in enumerate(self.u_data):
            new_data = copy.deepcopy(data)
            new_data[3] = label[idx]
            new_train_data.append(new_data)

        num_after_merge = len(np.unique(np.array(label)))
        print("num of label before merge: ", num_before_merge, " after_merge: ", num_after_merge, " sub: ",
              num_before_merge - num_after_merge)
        return new_train_data, label

    def generate_average_feature(self, labels):
        # extract feature/classifier
        u_feas, fcs = self.get_feature(self.u_data)

        # images of the same cluster
        label_to_images = {}
        for idx, l in enumerate(labels):
            label_to_images[l] = label_to_images.get(l, []) + [idx]

        # calculate average feature/classifier of a cluster
        feature_avg = np.zeros((len(label_to_images), len(u_feas[0])))
        fc_avg = np.zeros((len(label_to_images), len(fcs[0])))
        for l in label_to_images:
            feas = u_feas[label_to_images[l]]
            feature_avg[l] = np.mean(feas, axis=0)
            fc_avg[l] = np.mean(fcs[label_to_images[l]], axis=0)
        return u_feas, feature_avg, label_to_images, fc_avg

