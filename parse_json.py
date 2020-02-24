import json
import numpy as np
import random
import os
import argparse
from utils import *


def RL_choose_by_prob(data, node=4):
    val_acc = data['val_acc']
    print(val_acc)
    samples = data['sample']
    optim_prob = data['optim_prob']
    
    # remove the invalid value
    val_acc[0] = 0
    best_val_acc = max(val_acc)
    index = val_acc.index(best_val_acc)
    prob = np.max(np.array(optim_prob[index]), axis=1).reshape(2, -1)
    sample = np.array(samples[index]).reshape(2, -1)
    
    mask = []
    for i in range(node):
        for j in range(i+2):
            mask.append(i)
    mask = np.array(mask)
    
    new_sample = np.zeros_like(prob) + 7
    for i, c in enumerate(['normal', 'reduce']):
        prob_c = prob[i, :]
        for j in range(node):
            prob_c_masked = np.int8(mask == j) * prob_c
            for keep in range(2):
                ind = np.argmax(prob_c_masked)
                new_sample[i][ind] = sample[i][ind]
                prob_c_masked[ind] = 0
    new_sample = np.int8(new_sample).reshape(1, -1).tolist()[0]
    new_geno = convert_sample_to_genotype(new_sample)
    
    print('# index: {}'.format(index))
    print('# sample: {}'.format(samples[index]))
    print('# best genotype: {}'.format(data['best_genotype']))
    print('# new sample: {}'.format(new_sample))
    print('# new genotype:')
    print(new_geno)
    
    return new_geno


def Evolution_choose_randomly(data, node=4, number=10):
    val_acc = data['val_acc']
    samples = data['sample']
    best_val_acc = max(val_acc)
    index = val_acc.index(best_val_acc) # index of net with heightest acc
    sample = np.array(samples[index]) # best sample
    
    def get_mask_for_cell(node):
        mask = []
        for i in range(node):
            mask_n = [1 for n in range(i + 2)] # for each node
            index_list = list(range(i + 2)) # index of mask_n
            for j in range(i):
                ind = random.sample(index_list, 1)[0]
                mask_n[ind] = 0 # remove this edge
                index_list.remove(ind) # remove the one has been choosen
            mask.extend(mask_n)
        return mask
    
    new_genotypes = []
    print('# index: {}'.format(index))
    print('# sample: {}'.format(sample.tolist()))
    print('# best acc: {}'.format(best_val_acc))
    print('# best genotype: {}'.format(data['best_genotype']))
    
    for num in range(number):
        mask = get_mask_for_cell(node)
        mask.extend(get_mask_for_cell(node))
        inverse_mask = np.int8([0 if m == 1 else 1 for m in mask]) * 7
        mask = np.int8(mask)
        new_sample = sample * mask + inverse_mask
        new_geno = convert_sample_to_genotype(new_sample.tolist())
        new_genotypes.append(new_geno)
        
        print('# sample: {}'.format(sample.tolist()))
        print('# keep mask: {}'.format(mask.tolist()))
        print('# new_sample: {}'.format(new_sample.tolist()))
        print('# new genotype:')
        print(new_geno)
        
    return new_genotypes


def plt_reward_of_RL(logger):
    reward = []
    
    if not os.access(logger, os.R_OK):
        raise ValueError('%s not exists' % logger)
    with open(logger, 'r') as f:
        data = f.readlines()
        for line in data:
            if 'baseline reward' in line:
                reward.append(float(line.strip()[-7:-1]))
    if len(reward) == 0:
        raise ValueError('key word not found')
    step = list(range(len(reward)))
    try: 
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        
        plt.plot(step, reward)
        s = os.path.basename(logger).split('.')[0].split('_')
        saveto = 'experiment/RL/reward_{}_{}.png'.format(s[1], s[2])
        plt.savefig(saveto, format='png', dip=300)
    except:
        raise ValueError('can not plot figs')
    
    return reward

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('ParseJson')
    parser.add_argument('--method', default='RL', type=str, choices=['RL', 'EA'], required=True)
    parser.add_argument('--param', choices=['BPE1', 'BPE2'], type=str, required=True)
    parser.add_argument('--run_id', default=0, type=int, required=True)
    parser.add_argument('--saveto', default='experiment/EA/sampling.txt', type=str)
    args = parser.parse_args()


    if args.method == 'RL':
        path = 'experiment/RL/run_{}_{}.json'.format(args.param, args.run_id)
        data = json.load(open(path, 'r'))
        best_geno_rl = RL_choose_by_prob(data)
        rewards = plt_reward_of_RL(path.replace('.json', '.log').replace('run', 'RL'))
    elif args.method == 'EA':
        path = 'experiment/EA/run_{}_{}.json'.format(args.param, args.run_id)
        data = json.load(open(path, 'r'))
        random_geno_evo = Evolution_choose_randomly(data, number=10)
        
        with open(args.saveto, 'w') as f:
            for geno in random_geno_evo:
                f.write(geno+'\n')
    else:
        raise ValueError('Error method')
