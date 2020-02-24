import argparse
import sys
import genotypes as gt
import numpy as np
import utils

def generate_random_structure(node = 4, k =2):
    total_edge = sum(list(range(2, node + 2))) * 2
    cell_edge = int(total_edge / 2)
    num_ops = len(gt.PRIMITIVES)
    weight = np.random.randn(total_edge, num_ops)
    theta_norm = utils.darts_weight_unpack(weight[0:cell_edge], node)
    theta_reduce = utils.darts_weight_unpack(weight[cell_edge:], node)
    gene_normal = gt.parse_numpy(theta_norm, k=k)
    gene_reduce = gt.parse_numpy(theta_reduce, k=k)
    concat = range(2, 2+node)
    return gt.Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Generator')
    parser.add_argument('--num', default=100, type=int, help='number of achitectures to generate')
    parser.add_argument('--saveto', default='random_darts_architecture.txt', type=str)
    args = parser.parse_args()
    
    file = open(args.saveto, 'w+')
    for i in range(args.num):
        graph = generate_random_structure()
        file.write(str(graph)+'\n')
    file.close()