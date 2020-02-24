""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import preproc
from genotypes import *

def get_data(dataset, data_path, input_size, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, input_size, cutout_length)
    trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    shape = trn_data.train_data.shape
    
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    ret = [input_size, input_channels, n_classes, trn_data]
    if validation: # append validation data
        ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))

    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def convert_sample_to_genotype(array, l=28):
    """
    array is an array with shape [1, 28]
    return the Genotype of sample
    """
    if isinstance(array, str):
        array = np.array(array.split())
        assert len(array) == l
    elif isinstance(array, list):
        assert len(array) == l
        array = np.array(array)
    else:
        raise ValueError('error type of sample')

    index = []  # index of input nodes for current node
                # None ops if index is 7
    for i in range(4):
        index.extend([j for j in range(i+2)])

    geno_normal = []
    geno_reduce = []
    ops_normal = array[0:l//2]
    ops_reduce = array[l//2:]

    flag = 0
    for node in range(4):
        geno_curr = []
        op_ind = ops_normal[flag:flag+node+2]
        in_ind = index[flag:flag+node+2]
        for op, in_node in zip(op_ind, in_ind):
            if int(op) == 7: continue
            geno_curr.append(('%s' % PRIMITIVES[int(op)], in_node))
        flag = flag+node+2
        geno_normal.append(geno_curr)

    flag = 0
    for node in range(4):
        geno_curr = []
        op_ind = ops_reduce[flag:flag+node+2]
        in_ind = index[flag:flag+node+2]
        for op, in_node in zip(op_ind, in_ind):
            if int(op) == 7: continue
            geno_curr.append(('{}'.format(PRIMITIVES[int(op)]), in_node))
        flag = flag+node+2
        geno_reduce.append(geno_curr)

    genotype_str = "Genotype(normal=[{}, {}, {}, {}], normal_concat=range(2, 6), reduce=[{}, {}, {}, {}], reduce_concat=range(2, 6))".format(geno_normal[0], \
                    geno_normal[1], geno_normal[2], geno_normal[3], geno_reduce[0], geno_reduce[1], \
                    geno_reduce[2], geno_reduce[3])
    
    return genotype_str


def darts_weight_unpack(weight, n_nodes):
    w_dag = []
    start_index = 0
    end_index = 2
    for i in range(n_nodes):
        w_dag.append(weight[start_index:end_index])
        start_index = end_index
        end_index += 3 + i
    return w_dag


def one_hot_to_index(one_hot_matrix):
    return np.array([np.where(r == 1)[0][0] for r in one_hot_matrix])


def index_to_one_hot(index_vector, C):
    return np.eye(C)[index_vector.reshape(-1)]


def netParams(model):
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

