import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from copy import deepcopy
import torch.nn.functional as F



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PN():
    def __init__(self, args, opt, model, tensorboard, source_dataloaders, target_dataloader):
        self.device = device

        self.args = args
        self.opt = opt
        self.tensorboard = tensorboard

        # init dataloader
        self.source_dataloaders = source_dataloaders
        self.target_dataloader = target_dataloader

        self.net = model.Extractor(opt).to(device)

        # init criterions
        self.class_criterion = nn.NLLLoss()

        self.target_support_set = next(iter(target_dataloader['train']))

        self.iters_spt = [iter(self.source_dataloaders[i]['train']) for i in range(len(self.source_dataloaders))]
        self.iters_qry = [iter(self.source_dataloaders[i]['test']) for i in range(len(self.source_dataloaders))]

        # init self.optimizer
        self.optimizer = optim.Adam([{'params': self.net.parameters()}], lr=opt['learning_rate'],
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

    def load_checkpoint(self, checkpoint_path):
        path = checkpoint_path
        checkpoint = torch.load(path)

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

    def euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: N (minibatch size) x D
        # y: M (num class) x D. prototypes
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

    def get_prototypes(self, feature_extractor, data, label):

        feature_spt = feature_extractor(data)

        prototpyes = torch.FloatTensor(self.opt['num_class'], feature_spt.shape[-1]).to(device) # make FloatTensor with shape num_classes x F-dim1 x F-dim2...
        for i in range(self.opt['num_class']):
            prototpyes[i] = feature_spt[(label == i).nonzero().view(-1)].mean(0)

        self.prototypes = prototpyes

    def get_loss_and_confusion_matrix_for_query(self, feature_extractor, criterion, data, label):

        feature_data = feature_extractor(data)
        feature_data = feature_data.view(feature_data.shape[0],-1) # KN, 30720

        dists = self.euclidean_dist(feature_data, self.prototypes)
        preds_of_data = F.log_softmax(-dists, dim=1)
        loss_of_data = criterion(preds_of_data, label)
        pred_label = preds_of_data.max(1, keepdim=False)[1]

        labels = [i for i in range(len(self.opt['classes']))]
        cm = confusion_matrix(label.cpu().numpy(), pred_label.cpu().numpy(), labels=labels)

        return loss_of_data, cm

    def log_loss_results(self, condition, epoch, loss_avg, nshot=5):

        self.tensorboard.log_scalar(condition + '/loss_sum_' + str(nshot) + 'shot', loss_avg, epoch)

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

    def generate_random_tasks(self, num_repeat = 1):

        # generate synthetic domain

        with torch.no_grad():
            support_set_from_domains = []
            query_set_from_domains = []
            weights_per_domain = []

            for domain_i in range(len(self.source_dataloaders)): # for each task

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
                
                for _ in range(len(self.source_dataloaders[domain_i]['train'])): # append domain_i as many as the number of the domain data
                    weights_per_domain.append(domain_i)

            synthetic_supports = [] # list of (K shot x Num classes x dim1 x dim2, Kshot x NC, K shot x NC)
            synthetic_queries = []

            for i in range(num_repeat):
                selected_domain_indices = np.random.choice(weights_per_domain, self.opt['num_class']) # generate random indices to choose domain per class
                tmp_spt_feat = torch.FloatTensor(support_set_from_domains[0][0].shape[0], 0, *(support_set_from_domains[0][0].shape[2:]))
                tmp_spt_cl = torch.LongTensor(support_set_from_domains[0][1].shape[0], 0)
                tmp_spt_dl = torch.LongTensor(support_set_from_domains[0][2].shape[0], 0)

                tmp_qry_feat = torch.FloatTensor(query_set_from_domains[0][0].shape[0], 0, *(support_set_from_domains[0][0].shape[2:]))
                tmp_qry_cl = torch.LongTensor(query_set_from_domains[0][1].shape[0], 0)
                tmp_qry_dl = torch.LongTensor(query_set_from_domains[0][2].shape[0], 0)


                for class_id, domain_id in enumerate(selected_domain_indices):
                    class_id = np.random.randint(self.opt['num_class']) # set a random class id
                    class_id = torch.tensor([class_id])

                    ## for support set
                    feature, class_label, domain_label = support_set_from_domains[domain_id]

                    #select class dimension
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
        num_task = len(self.source_dataloaders)

        self.net.train()

        losses_q = 0.0
        confusion_matrices = np.zeros((self.opt['num_class'], self.opt['num_class']),dtype=int)
        self.reset_layer()


        synthetic_supports, synthetic_queries = self.generate_random_tasks(num_repeat=num_task)



        for task_i in range(num_task): # for each task

            support_batch = synthetic_supports[task_i]
            query_batch = synthetic_queries[task_i]

            support_set =  support_batch
            query_set = query_batch

            for k in range(self.args.nshot): # give random class number of increasing order
                for i in range(self.opt['num_class']):
                    support_set[1][k][i] = i
                    query_set[1][k][i] = i


            support_set[0] = support_set[0].view(-1, *(support_set[0].shape[2:]))# K shot x Num classes x dim1 x dim2 ... => KN x dim1 x dim2 ...
            support_set[1] = support_set[1].view(-1)
            support_set[2] = support_set[2].view(-1)


            query_set[0] = query_set[0].view(-1,*(query_set[0].shape[2:]))
            query_set[1] = query_set[1].view(-1)
            query_set[2] = query_set[2].view(-1)

            input_labeled_spt, class_label_labeled_spt, _ = self.get_label_and_data(support_set)
            input_labeled_qry, class_label_labeled_qry, _ = self.get_label_and_data(query_set)

            self.get_prototypes(self.net, input_labeled_spt, class_label_labeled_spt)

            class_loss_query, cm_query = self.get_loss_and_confusion_matrix_for_query(self.net,
                                                                                      self.class_criterion,
                                                                                      input_labeled_qry,
                                                                                      class_label_labeled_qry)

            losses_q += class_loss_query

            with torch.no_grad():
                confusion_matrices = confusion_matrices + cm_query

        loss_query = losses_q / num_task

        self.optimizer.zero_grad()
        loss_query.backward()
        self.optimizer.step()

        losses_q = loss_query.item()
        accs = 100 * np.sum(np.diagonal(confusion_matrices))/np.sum(confusion_matrices)

        np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
        print('###############################################################')
        print('[Train] epoch:' + str(epoch) + '\tloss:' + str(losses_q))
        print('[Train] epoch:' + str(epoch) + '\taccs:' + str(accs))

        self.log_loss_results('train', epoch=epoch, loss_avg=loss_query)
        self.log_accuracy_results('test',
                                  'labeled',
                                  epoch,
                                  cm_class=confusion_matrices)
        return


    def evaluation(self, epoch, condition, nshot=5):

        confusion_matrices = np.zeros((self.opt['num_class'], self.opt['num_class']),dtype=int)
        self.reset_layer()

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        net.train()
        losses_q = 0.0

        support_set = [x[:nshot].data.clone() for x in self.target_support_set]  # evaluation with only nshots
        support_set[0] = support_set[0].view(-1, *(support_set[0].shape[2:]))
        support_set[1] = support_set[1].view(-1)
        support_set[2] = support_set[2].view(-1)

        for batch_idx, query_set in tqdm(enumerate(self.target_dataloader[condition]), total=len(self.target_dataloader[condition])):

            query_set[0] = query_set[0].view(-1, *(query_set[0].shape[2:]))
            query_set[1] = query_set[1].view(-1)
            query_set[2] = query_set[2].view(-1)

            input_labeled_spt, class_label_labeled_spt, _ = self.get_label_and_data(support_set)
            input_labeled_qry, class_label_labeled_qry, _ = self.get_label_and_data(query_set)

            self.get_prototypes(self.net, input_labeled_spt, class_label_labeled_spt)

            class_loss_query, cm_query = self.get_loss_and_confusion_matrix_for_query(self.net,
                                                                                      self.class_criterion,
                                                                                      input_labeled_qry,
                                                                                      class_label_labeled_qry)

            with torch.no_grad():
                confusion_matrices = confusion_matrices + cm_query
                losses_q += class_loss_query

        accs = 100 * np.sum(np.diagonal(confusion_matrices))/np.sum(confusion_matrices)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        print('---------------------------------------------------')
        print('[{:s}] epoch:{:d} nshot:{:d}'.format(condition, epoch, nshot) + '\tloss:' + str(losses_q))
        print('[{:s}] epoch:{:d} nshot:{:d}'.format(condition, epoch, nshot) + '\taccs:' + str(accs))

        self.log_loss_results(condition, epoch=epoch, loss_avg=losses_q)
        self.log_accuracy_results(condition,
                                  'test',
                                  epoch,
                                  cm_class=confusion_matrices, nshot=nshot)
        return accs, losses_q


    def validation(self, epoch):
        """
        Validate the performance of the model
        """
        class_accuracy_of_test_data, loss = self.evaluation(epoch, 'valid', nshot=5)

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
