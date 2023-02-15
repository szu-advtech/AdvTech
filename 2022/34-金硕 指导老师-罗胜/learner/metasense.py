import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from copy import deepcopy



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaSense():
    def __init__(self, args, opt, model, tensorboard, source_dataloaders, target_dataloader, lr, is_shuffle_label=False,
                 is_per_cond_task_only=False):
        self.device = device

        self.args = args
        self.opt = opt
        self.tensorboard = tensorboard

        # init dataloader
        self.source_dataloaders = source_dataloaders
        self.target_dataloader = target_dataloader

        self.net = model.MAML_BN(opt).to(device)

        self.task_lr = lr
        self.meta_lr = opt['learning_rate']
        self.update_step_train = 5
        self.update_step_test = 10
        self.is_shuffle_label = is_shuffle_label
        self.is_per_cond_task_only = is_per_cond_task_only
        # init criterions
        self.class_criterion = nn.NLLLoss()


        self.target_support_set = next(iter(target_dataloader['train']))

        self.iters_spt = [iter(self.source_dataloaders[i]['train']) for i in range(len(self.source_dataloaders))]
        self.iters_qry = [iter(self.source_dataloaders[i]['test']) for i in range(len(self.source_dataloaders))]

        # init self.optimizer
        self.optimizer = optim.Adam([{'params': self.net.parameters()}], lr=self.meta_lr,
                                    weight_decay=opt['weight_decay'])


    def save_checkpoint(self, epoch, epoch_acc, best_acc, checkpoint_path):
        state = {}
        state['epoch'] = epoch
        state['epoch_acc'] = epoch_acc
        state['best_acc'] = best_acc
        state['class_classifier'] = self.net.state_dict()
        state['optimizer'] = self.optimizer.state_dict()

        torch.save(state, checkpoint_path)

        return

    def load_checkpoint(self, checkpoint_path, from_other_architecture=False):
        path = checkpoint_path
        checkpoint = torch.load(path)

        if from_other_architecture:
            self.net.load_checkpoint(checkpoint)

        else:
            self.net.load_state_dict(checkpoint['class_classifier'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint

    def reset_layer(self):
        # Remember that Pytorch accumulates gradients.x
        # We need to clear them out before each instance
        # that means we have to call optimizer.zero_grad() before each backward.
        self.net.zero_grad()

    def get_label_and_data(self, data):
        input_of_data, class_label_of_data, domain_label_of_data = data
        input_of_data = input_of_data.to(device)
        class_label_of_data = class_label_of_data.to(device)
        domain_label_of_data = domain_label_of_data.to(device)

        return input_of_data, class_label_of_data, domain_label_of_data

    def get_loss_and_confusion_matrix(self, classifier, params, criterion, data, label):

        preds_of_data = classifier(data, params)
        pred_label = preds_of_data.max(1, keepdim=False)[1]
        loss_of_data = criterion(preds_of_data, label)

        labels = [i for i in range(len(self.opt['classes']))]
        cm = confusion_matrix(label.cpu().numpy(), pred_label.cpu().numpy(), labels=labels)

        return loss_of_data, cm

    def log_loss_results(self, condition, epoch, loss_avg, nshot=5):

        self.tensorboard.log_scalar(condition + '/loss_sum_'+ str(nshot) + 'shot', loss_avg, epoch)

        return loss_avg

    def log_accuracy_results(self, condition, suffix, epoch, cm_class, nshot = 5):

        assert (condition in ['valid', 'test'])
        assert (suffix in ['labeled', 'unlabeled', 'test'])

        class_accuracy = 100.0 * np.sum(np.diagonal(cm_class)) / np.sum(cm_class)
        self.tensorboard.log_scalar(condition + '/' + 'accuracy_class_' + suffix + '_' + str(nshot) + 'shot',
                                    class_accuracy, epoch)

        self.tensorboard.log_confusion_matrix(condition + '_accuracy_class_' + suffix + '_' + str(nshot) + 'shot',
                                              cm_class,
                                              self.opt['classes'], epoch)
        return class_accuracy


    def generate_multi_cond_tasks(self, num_repeat=1):

        with torch.no_grad():
            # generate synthetic domain
            support_set_from_domains = []
            query_set_from_domains = []

            for domain_i in range(len(self.source_dataloaders)):  # for each task

                try:
                    train_batch_i = next(self.iters_spt[domain_i])
                except StopIteration:
                    self.iters_spt[domain_i] = iter(self.source_dataloaders[domain_i]['train'])
                    train_batch_i = next(self.iters_spt[domain_i])

                try:
                    test_batch_i = next(self.iters_qry[domain_i])
                except StopIteration:
                    self.iters_qry[domain_i] = iter(self.source_dataloaders[domain_i]['test'])
                    test_batch_i = next(self.iters_qry[domain_i])


                support_set_from_domains.append(train_batch_i)
                query_set_from_domains.append(test_batch_i)

            synthetic_supports = []  # list of (K shot x Num classes x dim1 x dim2, Kshot x NC, K shot x NC)
            synthetic_queries = []

            for i in range(num_repeat):
                selected_domain_indices = np.random.randint(len(self.source_dataloaders), size=self.opt[
                    'num_class'])  # generate random indices to choose domain per class

                tmp_spt_feat = torch.FloatTensor(support_set_from_domains[0][0].shape[0], 0,
                                                 *(support_set_from_domains[0][0].shape[2:]))
                tmp_spt_cl = torch.LongTensor(support_set_from_domains[0][1].shape[0], 0)
                tmp_spt_dl = torch.LongTensor(support_set_from_domains[0][2].shape[0], 0)

                tmp_qry_feat = torch.FloatTensor(query_set_from_domains[0][0].shape[0], 0,
                                                 *(support_set_from_domains[0][0].shape[2:]))
                tmp_qry_cl = torch.LongTensor(query_set_from_domains[0][1].shape[0], 0)
                tmp_qry_dl = torch.LongTensor(query_set_from_domains[0][2].shape[0], 0)

                for class_id, domain_id in enumerate(selected_domain_indices):

                    class_id = torch.tensor([class_id])

                    ## for support set
                    feature, class_label, domain_label = support_set_from_domains[domain_id]

                    # select class dimension
                    indices = class_label == class_id
                    feature = feature[indices].view(feature.shape[0],1,*feature.shape[2:])
                    class_label = class_label[indices].view(-1,1)
                    domain_label = domain_label[indices].view(-1,1)

                    tmp_spt_feat = torch.cat((tmp_spt_feat, feature), dim=1)
                    tmp_spt_cl = torch.cat((tmp_spt_cl, class_label), dim=1)
                    tmp_spt_dl = torch.cat((tmp_spt_dl, domain_label), dim=1)

                    ## for query set
                    feature, class_label, domain_label = query_set_from_domains[domain_id]

                    # select class dimension
                    indices = class_label == class_id
                    feature = feature[indices].view(feature.shape[0],1,*feature.shape[2:])
                    class_label = class_label[indices].view(-1,1)
                    domain_label = domain_label[indices].view(-1,1)

                    tmp_qry_feat = torch.cat((tmp_qry_feat, feature), dim=1)
                    tmp_qry_cl = torch.cat((tmp_qry_cl, class_label), dim=1)
                    tmp_qry_dl = torch.cat((tmp_qry_dl, domain_label), dim=1)

                synthetic_supports.append([tmp_spt_feat, tmp_spt_cl, tmp_spt_dl])
                synthetic_queries.append([tmp_qry_feat, tmp_qry_cl, tmp_qry_dl])

        return synthetic_supports, synthetic_queries

    def train(self, epoch):
        """
        Train the model
        """

        # setup models
        if self.is_per_cond_task_only:
            self.num_synthetic_domains = 0
        else:
            self.num_synthetic_domains = len(self.source_dataloaders)
        num_task = len(self.source_dataloaders) + self.num_synthetic_domains

        self.net.train()

        losses_q = [0.0 for _ in range(self.update_step_train + 1)]  # losses_q[i] is the loss on step i
        losses_q_synthetic = [0.0 for _ in range(self.update_step_train + 1)]  # losses_q[i] is the loss on step i

        confusion_matrices = [np.zeros((self.opt['num_class'], self.opt['num_class']), dtype=int) for _ in
                              range(self.update_step_train + 1)]
        confusion_matrices_synthetic = [np.zeros((self.opt['num_class'], self.opt['num_class']), dtype=int) for _ in
                                        range(self.update_step_train + 1)]
        self.reset_layer()

        synthetic_supports, synthetic_queries = self.generate_multi_cond_tasks(num_repeat=self.num_synthetic_domains)


        for task_i in range(num_task):  # for each task

            if task_i < len(self.source_dataloaders): # per-condition tasks
                domain_i = task_i
                try:
                    train_batch_i = next(self.iters_spt[domain_i])
                except StopIteration:
                    self.iters_spt[domain_i] = iter(self.source_dataloaders[domain_i]['train'])
                    train_batch_i = next(self.iters_spt[domain_i])

                try:
                    test_batch_i = next(self.iters_qry[domain_i])
                except StopIteration:
                    self.iters_qry[domain_i] = iter(self.source_dataloaders[domain_i]['test'])
                    test_batch_i = next(self.iters_qry[domain_i])


                support_batch = train_batch_i
                query_batch = test_batch_i
            else: # multi-conditioned tasks
                support_batch = synthetic_supports[task_i - len(self.source_dataloaders)]
                query_batch = synthetic_queries[task_i - len(self.source_dataloaders)]

            support_set = support_batch
            support_set[0] = support_set[0].view(-1, *(
                support_set[0].shape[2:]))  # K shot x Num classes x dim1 x dim2 ... => KN x dim1 x dim2 ...
            support_set[1] = support_set[1].view(-1)
            support_set[2] = support_set[2].view(-1)

            query_set = query_batch
            query_set[0] = query_set[0].view(-1, *(query_set[0].shape[2:]))
            query_set[1] = query_set[1].view(-1)
            query_set[2] = query_set[2].view(-1)

            if self.is_shuffle_label:
                shuffled_labels = list(range(self.opt['num_class']))
                np.random.shuffle(shuffled_labels)
                for i in range(len(support_set[1])):
                    support_set[1][i] = shuffled_labels[support_set[1][i]]
                for i in range(len(query_set[1])):
                    query_set[1][i] = shuffled_labels[query_set[1][i]]

            input_labeled_spt, class_label_labeled_spt, _ = self.get_label_and_data(support_set)
            input_labeled_qry, class_label_labeled_qry, _ = self.get_label_and_data(query_set)

            fast_weights = self.net.parameters()

            for k in range(0, self.update_step_train + 1):

                class_loss_query, cm_query = self.get_loss_and_confusion_matrix(self.net, fast_weights,
                                                                                self.class_criterion,
                                                                                input_labeled_qry,
                                                                                class_label_labeled_qry)
                if k == self.update_step_train:
                    if task_i < len(self.source_dataloaders):
                        losses_q[k] += class_loss_query
                    else:
                        losses_q_synthetic[k] += class_loss_query
                else:
                    if task_i < len(self.source_dataloaders):
                        losses_q[k] += float(class_loss_query)
                    else:
                        losses_q_synthetic[k] += float(class_loss_query)

                with torch.no_grad():

                    if task_i < len(self.source_dataloaders):
                        confusion_matrices[k] = confusion_matrices[k] + cm_query
                    else:
                        confusion_matrices_synthetic[k] = confusion_matrices_synthetic[k] + cm_query

                if k != self.update_step_train:  # ignore the last unnecessary update
                    # 1. run the i-th task and compute loss
                    class_loss_support, _ = self.get_loss_and_confusion_matrix(self.net, fast_weights,
                                                                               self.class_criterion,
                                                                               input_labeled_spt,
                                                                               class_label_labeled_spt)
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(class_loss_support, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, fast_weights)))

        loss_query = (losses_q[-1] + losses_q_synthetic[-1]) / num_task

        self.optimizer.zero_grad()
        loss_query.backward()
        self.optimizer.step()

        losses_q = np.array([loss if isinstance(loss, float) else loss.item() for loss in losses_q]) / num_task
        losses_q_synthetic = np.array(
            [loss if isinstance(loss, float) else loss.item() for loss in losses_q_synthetic]) / num_task
        losses_sum = np.add(losses_q_synthetic, losses_q)
        losses_sum = np.array([loss if isinstance(loss, float) else loss.item() for loss in losses_sum])

        accs = np.array([100 * np.sum(np.diagonal(correct)) / np.sum(correct) for correct in confusion_matrices])
        accs_synthetic = np.array(
            [100 * np.sum(np.diagonal(correct)) / np.sum(correct) for correct in confusion_matrices_synthetic])
        accs_sum = np.add(confusion_matrices, confusion_matrices_synthetic)
        accs_sum = np.array([100 * np.sum(np.diagonal(correct)) / np.sum(correct) for correct in accs_sum])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        print('###############################################################')
        print('[Train] epoch:' + str(epoch) + '\tloss qry:' + str(losses_q))
        print('[Train] epoch:' + str(epoch) + '\tloss syn:' + str(losses_q_synthetic))
        print('[Train] epoch:' + str(epoch) + '\tloss sum:' + str(losses_sum))
        print('[Train] epoch:' + str(epoch) + '\taccs qry:' + str(accs))
        print('[Train] epoch:' + str(epoch) + '\taccs syn:' + str(accs_synthetic))
        print('[Train] epoch:' + str(epoch) + '\taccs sum:' + str(accs_sum))

        self.log_loss_results('train', epoch=epoch, loss_avg=loss_query)
        self.log_accuracy_results('test',
                                  'labeled',
                                  epoch,
                                  cm_class=confusion_matrices[-1])

    def evaluation(self, epoch, condition, nshot=5):

        num_task = 1

        confusion_matrices = [np.zeros((self.opt['num_class'], self.opt['num_class']), dtype=int) for _ in
                              range(self.update_step_test + 1)]

        losses_q = [0.0 for _ in range(self.update_step_test + 1)]  # losses_q[i] is the loss on step i

        self.reset_layer()

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        net.train()


        support_set = [x[:nshot].data.clone() for x in self.target_support_set]  # evaluation with only nshots
        support_set[0] = support_set[0].view(-1, *(support_set[0].shape[2:]))
        support_set[1] = support_set[1].view(-1)
        support_set[2] = support_set[2].view(-1)

        for batch_idx, query_set in tqdm(enumerate(self.target_dataloader[condition]),
                                         total=len(self.target_dataloader[condition])):

            query_set[0] = query_set[0].view(-1, *(query_set[0].shape[2:]))
            query_set[1] = query_set[1].view(-1)
            query_set[2] = query_set[2].view(-1)

            input_labeled_spt, class_label_labeled_spt, _ = self.get_label_and_data(support_set)
            input_labeled_qry, class_label_labeled_qry, _ = self.get_label_and_data(query_set)

            fast_weights = net.parameters()
            for k in range(0, self.update_step_test + 1):
                class_loss_query, cm_query = self.get_loss_and_confusion_matrix(net, fast_weights,
                                                                                self.class_criterion,
                                                                                input_labeled_qry,
                                                                                class_label_labeled_qry)

                with torch.no_grad():
                    losses_q[k] += float(class_loss_query)
                    confusion_matrices[k] = confusion_matrices[k] + cm_query

                if k != self.update_step_test:
                    # 1. run the i-th task and compute loss for k=1~K-1
                    class_loss_support, _ = self.get_loss_and_confusion_matrix(net, fast_weights,
                                                                               self.class_criterion,
                                                                               input_labeled_spt,
                                                                               class_label_labeled_spt)
                    # 2. compute grad on theta_pi
                    grad = torch.autograd.grad(class_loss_support, fast_weights)
                    # 3. theta_pi = theta_pi - train_lr * grad
                    fast_weights = list(map(lambda p: p[1] - self.task_lr * p[0], zip(grad, fast_weights)))

        loss_query = losses_q[-1] / num_task
        losses_q = np.array([loss if isinstance(loss, float) else loss.item() for loss in losses_q]) / num_task
        accs = np.array([100 * np.sum(np.diagonal(correct)) / np.sum(correct) for correct in confusion_matrices])
        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

        print('---------------------------------------------------')
        print('[{:s}] epoch:{:d} nshot:{:d}'.format(condition, epoch, nshot) + '\tloss:' + str(losses_q))
        print('[{:s}] epoch:{:d} nshot:{:d}'.format(condition, epoch, nshot) + '\taccs:' + str(accs))

        self.log_loss_results(condition, epoch=epoch, loss_avg=loss_query)
        self.log_accuracy_results(condition,
                                  'test',
                                  epoch,
                                  cm_class=confusion_matrices[-1], nshot=nshot)

        return accs[-1], losses_q

    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss = self.evaluation(epoch, 'valid', nshot=self.args.nshot)

        # setup the network
        return class_accuracy_of_test_data, loss

    def test(self, epoch):
        """
        Test the performance of the model
        """
        class_accuracy_of_test_data = []
        for i in range(10):
            accuracy_for_shot, loss_fot_shot = self.evaluation(epoch, 'test', nshot= i + 1)
            class_accuracy_of_test_data.append(accuracy_for_shot)

        return class_accuracy_of_test_data
