import torch
import torch.nn as nn

def has_optim_in_children(subnet):
    '''
    check if there is specific optim parameters in a subnet.
    :param subnet:
    :return:
    '''
    label = False
    for module in subnet.children():
        if hasattr(module, 'optim_spec') and module.optim_spec:
            label = True
            break
        else:
            label = has_optim_in_children(module)

    return label

def find_optim_module(net):
    '''
    classify modules in a net into has specific optim specs or not.
    :param net:
    :return:
    '''
    module_optim_pairs = []
    other_modules = []
    for module in net.children():
        if hasattr(module, 'optim_spec'):
            module_optim_pairs += [{'module':module, 'optim_spec':module.optim_spec}]
        elif not has_optim_in_children(module):
            other_modules += [module]
        else:
            module_optim_pairs += find_optim_module(module)[0]
            other_modules += find_optim_module(module)[1]

    return module_optim_pairs, other_modules

def load_scheduler(config, optimizer):
    '''
    get scheduler for optimizer.
    :param config: configuration file
    :param optimizer: torch optimizer
    :return:
    '''
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=config['scheduler']['milestones'],
                                                     gamma=config['scheduler']['gamma'])

    return scheduler

def load_bnm_scheduler(cfg, net, start_epoch):
    bn_lbmd = lambda it: max(cfg.config['bnscheduler']['bn_momentum_init'] * cfg.config['bnscheduler']['bn_decay_rate'] ** (
        int(it / cfg.config['bnscheduler']['bn_decay_step'])), cfg.config['bnscheduler']['bn_momentum_max'])
    bnm_scheduler = BNMomentumScheduler(cfg, net, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)
    return bnm_scheduler

def load_optimizer(config, net):
    '''
    get optimizer for networks
    :param config: configuration file
    :param model: nn.Module network
    :return:
    '''

    module_optim_pairs, other_modules = find_optim_module(net)
    default_optim_spec = config['optimizer']

    optim_params = []

    if config['optimizer']['method'] == 'Adam':
        '''collect parameters with specific optimizer spec'''
        for module in module_optim_pairs:
            optim_params.append({'params': filter(lambda p: p.requires_grad, module['module'].parameters()),
                                 'lr': float(module['optim_spec']['lr']),
                                 'betas': tuple(module['optim_spec']['betas']),
                                 'eps': float(module['optim_spec']['eps']),
                                 'weight_decay': float(module['optim_spec']['weight_decay'])})

        '''collect parameters with default optimizer spec'''
        other_params = list()
        for module in other_modules:
            other_params += list(module.parameters())

        optim_params.append({'params': filter(lambda p: p.requires_grad, other_params)})

        '''define optimizer'''
        optimizer = torch.optim.AdamW(optim_params,
                                      lr=float(default_optim_spec['lr']),
                                      betas=tuple(default_optim_spec['betas']),
                                      eps=float(default_optim_spec['eps']),
                                      weight_decay=float(default_optim_spec['weight_decay']))

    else:
        # use SGD optimizer.
        for module in module_optim_pairs:
            optim_params.append({'params': filter(lambda p: p.requires_grad, module['module'].parameters()),
                                 'lr': float(module['optim_spec']['lr'])})

        other_params = list()
        for module in other_modules:
            other_params += list(module.parameters())

        optim_params.append({'params': filter(lambda p: p.requires_grad, other_params)})
        optimizer = torch.optim.SGD(optim_params,
                                    lr=float(config['optimizer']['lr']),
                                    momentum=0.9)

    return optimizer

def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, cfg, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )
        self.cfg = cfg
        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def show_momentum(self):
        self.cfg.log_string('Current BN decay momentum :%f.' % (self.lmbd(self.last_epoch)))
