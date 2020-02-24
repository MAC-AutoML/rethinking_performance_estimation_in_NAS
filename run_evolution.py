import os
import random
import json
import argparse
import collections
import tensorflow as tf
import tensorflow_probability as tfp
import ConfigSpace
import numpy as np
import torch.nn as nn
import torchvision
import utils
from utils import *
from collections import namedtuple
from copy import deepcopy
from models.augment_cnn import AugmentCNN
from param_setting import *

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = ['max_pool_3x3', 'avg_pool_3x3', 'skip_connect', 'sep_conv_3x3', 
              'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5', 'none']


parser = argparse.ArgumentParser('EA')
parser.add_argument('--run_id', default=0, type=int, help='to identify the experiments')
parser.add_argument('--seed', default=2, type=int, help='random setting')
parser.add_argument('--param', type=str, choices=['BPE1', 'BPE2'], required=True, help='the hyperparameters for training')
parser.add_argument('--gpu_id', default=0, type=int, help='the id of gpu')
parser.add_argument('--output_path', default='experiment/RL', type=str, help='the path to save the results')
parser.add_argument('--data_path', default='data/', type=str, help='the path of data')
parser.add_argument('--n_iters', default=100, type=int, help='number of iterations for optimization method')
parser.add_argument('--pop_size', default=50, type=int, help='population size')
parser.add_argument('--sample_size', default=10, type=int, help='sample_size')
args = parser.parse_args()

# set device
device = torch.device("cuda")
torch.cuda.set_device(args.gpu_id)

# set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# shutdown cudnn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

config = param_BPE1 if args.param == 'BPE1' else param_BPE2
os.makedirs(args.output_path, exist_ok=True)
logger = utils.get_logger(os.path.join(args.output_path, 'EA_%s_%d.log' % (args.param, args.run_id)))



def train(train_loader, model, optimizer, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    def train_iter(X, y):
        N = X.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        loss += 0.4 * criterion(aux_logits, y)
        
        loss.backward()
        
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % 200 == 0 or step == len_train_loader - 1:
            logger.info("Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch + 1, config['epochs'], step, len_train_loader - 1, losses=losses, top1=top1, top5=top5))
            
    len_train_loader = len(train_loader)
    cur_step = epoch * len_train_loader
    cur_lr = optimizer.param_groups[0]['lr']

    model.train()
    for step, (X, y) in enumerate(train_loader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        train_iter(X, y)
        cur_step += 1
    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config['epochs'], top1.avg))


def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()
    len_val_loader = len(valid_loader)

    def val_iter(X, y):
        N = X.size(0)

        logits, _ = model(X)
        loss = criterion(logits, y)

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % 200 == 0 or step == len_val_loader - 1:
            logger.info("Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch + 1, config['epochs'], step, len_val_loader - 1, losses=losses, top1=top1, top5=top5))
    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            val_iter(X, y)  
    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config['epochs'], top1.avg))
    return top1.avg


def evaluation(sample, name):
    geno = eval(convert_sample_to_genotype(sample))
    logger.info('Model sample: {}'.format(sample))
    logger.info('Genotype: {}'.format(str(geno)))

    # get data with meta info    
    input_size, input_channels, n_classes, train_data, valid_data = utils.get_data(
        'cifar10', args.data_path, config['imagesize'], config['cutout'], validation=True)

    criterion = nn.CrossEntropyLoss().to(device)
    use_aux = True    
    
    # change size of input image 
    input_size = config['imagesize']
    
    model = AugmentCNN(input_size, input_channels, config['channel'], 10, config['layers'], True, geno)
    mb_params = utils.param_size(model)
    logger.info("Model size = {:.3f} MB".format(mb_params))
    model = nn.DataParallel(model, device_ids=[0]).to(device)
    
    # weights optimizer
    optimizer = torch.optim.SGD(model.parameters(), config['lr'], momentum=0.9, weight_decay=3e-4)

    # get data loader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batchsize'], \
                                               shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=config['batchsize'], \
                                               shuffle=True, num_workers=4, pin_memory=True)
    
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'])

    best_top1 = 0.
    len_train_loader = len(train_loader)
    
    # training loop
    for epoch in range(config['epochs']):
        lr_scheduler.step()
        drop_prob = 0.2 * epoch / config['epochs']
        model.module.drop_path_prob(drop_prob, config['fp'])

        train(train_loader, model, optimizer, criterion, epoch)

        cur_step = (epoch+1) * len_train_loader
        top1 = validate(valid_loader, model, criterion, epoch, cur_step)

        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        # utils.save_checkpoint(model, config.path, is_best)

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))
    return best_top1, geno


class NASCifar10(object):
    def __init__(self):
        self.val_acc = []
        self.best_acc = 0.0
        self.genotypes = []
        self.samples = []
        self.best_geno = ""

    def get_results(self):
        res = dict()
        res['val_acc'] = self.val_acc
        res['genotype'] = self.genotypes
        res['sample'] = self.samples
        res['best_val_acc'] = [self.best_acc]
        res['best_genotype'] = [self.best_geno]

        return res
    
    def objective_function(self, sample, name):
        if not isinstance(sample[0], int):
            sample = [s.numpy()[0] for s in sample]
            
        acc, geno = evaluation(sample, name)
        
        self.val_acc.append(float(acc))
        self.genotypes.append(str(geno))
        self.samples.append([int(s) for s in sample])
        if acc > self.best_acc:
            self.best_acc = float(acc)
            self.best_geno = str(geno)
            
        return 1 - acc
    
    def get_configuration_space(self):
        cs = ConfigSpace.ConfigurationSpace()
        OPS = PRIMITIVES[0:-1]
        for cell in ['normal', 'reduce']:
            for node in range(2, 6):
                for prev in range(0, node):
                    cs.add_hyperparameter(ConfigSpace.CategoricalHyperparameter("{}_{}_{}".format(cell, node, prev), OPS))
                    
        return cs


class Model(object):
    def __init__(self):
        self.arch = None
        self.accuracy = None

    def __str__(self):
        return '{0:b}'.format(self.arch)


def train_and_eval(config, name):
    error = b.objective_function(config, name)
    return 1 - error


def arch_to_sample(config):
    sample = []
    for cell in ['normal', 'reduce']:
        for node in range(2, 6):
            for prev in range(0, node):
                sample.append(int(PRIMITIVES.index(config['{}_{}_{}'.format(cell, node, prev)])))
    return sample


def random_architecture():
    config = cs.sample_configuration()
    return config


def mutate_arch(parent_arch):
    dim = np.random.randint(len(cs.get_hyperparameters()))
    hyper = cs.get_hyperparameters()[dim]
    
    if type(hyper) == ConfigSpace.OrdinalHyperparameter:
        choices = list(hyper.sequence)
    else:
        choices = list(hyper.choices)
    choices.remove(parent_arch[hyper.name])
    idx = np.random.randint(len(choices))
    child_arch = deepcopy(parent_arch)
    child_arch[hyper.name] = choices[idx]
    return child_arch


def regularized_evolution(cycles, population_size, sample_size):

    population = collections.deque()
    
    history = []

    logger.info('Initialize the population with random models')
    logger.info('population_size: {}'.format(population_size))
    pop_n = 0
    while len(population) < population_size:  # 100
        logger.info('population {}'.format(pop_n))
        model = Model()
        model.arch = random_architecture()
        model.accuracy = train_and_eval(arch_to_sample(model.arch), name='population_%03d' % pop_n)
        population.append(model)
        history.append(model)
        pop_n += 1
    logger.info('Initialized {} populations with random models'.format(population_size))

    iter_n = 0
    while len(history) < cycles:
        logger.info('Cycles %d' % iter_n)
        sample = []
        while len(sample) < sample_size:
            candidate = random.choice(list(population))
            sample.append(candidate)
        
        parent = max(sample, key=lambda i: i.accuracy)

        child = Model()
        child.arch = mutate_arch(parent.arch)
        child.accuracy = train_and_eval(arch_to_sample(child.arch), name='cycles_%03d' % iter_n)
        population.append(child)
        history.append(child)
        
        population.popleft()
        iter_n += 1

    return history

    
b = NASCifar10()
cs = b.get_configuration_space()

history = regularized_evolution(cycles=args.n_iters, 
                                population_size=args.pop_size, 
                                sample_size=args.sample_size)

res = b.get_results()

logger.info('Best accuracy: {}'.format(res['best_val_acc'][0]))
logger.info('Best accuracy: {}'.format(res['best_genotype'][0]))

fh = open(os.path.join(args.output_path, 'run_%s_%d.json' % (args.param, args.run_id)), 'w')
json.dump(res, fh)
fh.close()