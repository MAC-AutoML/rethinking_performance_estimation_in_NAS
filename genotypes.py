from collections import namedtuple
import torch
import torch.nn as nn
from models import ops


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'none'
]


# BPE models
BPE_models = {
    'EA_BPE1': "Genotype(normal=[[('avg_pool_3x3', 0), ('skip_connect', 1)], [('skip_connect', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 0), ('dil_conv_5x5', 1)], [('sep_conv_3x3', 0), ('sep_conv_5x5', 2)]], normal_concat=range(2, 6), reduce=[[('max_pool_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 0), ('skip_connect', 2)], [('max_pool_3x3', 1), ('skip_connect', 2)]], reduce_concat=range(2, 6))",

    'EA_BPE2': "Genotype(normal=[[('skip_connect', 0), ('avg_pool_3x3', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)], [('dil_conv_3x3', 1), ('avg_pool_3x3', 2)], [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)]], normal_concat=range(2, 6), reduce=[[('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], [('dil_conv_5x5', 0), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], [('skip_connect', 1), ('sep_conv_3x3', 4)]], reduce_concat=range(2, 6))",

    'RL_BPE1': "Genotype(normal=[[('skip_connect', 0), ('dil_conv_3x3', 1)], [('sep_conv_3x3', 1), ('avg_pool_3x3', 2)], [('sep_conv_3x3', 0), ('max_pool_3x3', 3)], [('sep_conv_5x5', 0), ('avg_pool_3x3', 3)]], normal_concat=range(2, 6), reduce=[[('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], [('sep_conv_5x5', 0), ('max_pool_3x3', 1)], [('dil_conv_3x3', 2), ('sep_conv_5x5', 3)], [('avg_pool_3x3', 0), ('sep_conv_3x3', 4)]], reduce_concat=range(2, 6))",

    'RL_BPE2': "Genotype(normal=[[('skip_connect', 0), ('avg_pool_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 0), ('skip_connect', 1)], [('avg_pool_3x3', 0), ('avg_pool_3x3', 2)]], normal_concat=range(2, 6), reduce=[[('sep_conv_5x5', 0), ('skip_connect', 1)], [('avg_pool_3x3', 1), ('sep_conv_3x3', 2)], [('dil_conv_5x5', 1), ('sep_conv_3x3', 3)], [('avg_pool_3x3', 1), ('max_pool_3x3', 4)]], reduce_concat=range(2, 6))",

    'DARTS_BPE1': "Genotype(normal=[[('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 4), ('sep_conv_5x5', 3)]], normal_concat=range(2, 6), reduce=[[('dil_conv_3x3', 0), ('sep_conv_5x5', 1)], [('sep_conv_5x5', 1), ('sep_conv_5x5', 0)], [('sep_conv_5x5', 3), ('dil_conv_5x5', 2)], [('dil_conv_5x5', 4), ('sep_conv_5x5', 3)]], reduce_concat=range(2, 6))",

    'DARTS_BPE2': "Genotype(normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], [('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], [('sep_conv_3x3', 0), ('sep_conv_5x5', 3)], [('sep_conv_3x3', 4), ('sep_conv_3x3', 0)]], normal_concat=range(2, 6), reduce=[[('dil_conv_5x5', 1), ('sep_conv_5x5', 0)], [('avg_pool_3x3', 0), ('sep_conv_5x5', 1)], [('dil_conv_5x5', 3), ('sep_conv_5x5', 2)], [('sep_conv_3x3', 2), ('sep_conv_3x3', 1)]], reduce_concat=range(2, 6))",

    'RS_BPE1': "Genotype(normal=[[('sep_conv_5x5', 0), ('dil_conv_3x3', 1)], [('max_pool_3x3', 0), ('sep_conv_3x3', 1)], [('skip_connect', 3), ('sep_conv_5x5', 0)], [('sep_conv_3x3', 2), ('skip_connect', 0)]], normal_concat=range(2, 6), reduce=[[('dil_conv_5x5', 1), ('max_pool_3x3', 0)], [('sep_conv_3x3', 1), ('max_pool_3x3', 0)], [('dil_conv_3x3', 0), ('max_pool_3x3', 1)], [('dil_conv_3x3', 2), ('sep_conv_3x3', 0)]], reduce_concat=range(2, 6))",

    'RS_BPE2': "Genotype(normal=[[('skip_connect', 1), ('sep_conv_3x3', 0)], [('avg_pool_3x3', 2), ('skip_connect', 0)], [('sep_conv_3x3', 1), ('avg_pool_3x3', 3)], [('avg_pool_3x3', 0), ('dil_conv_3x3', 2)]], normal_concat=range(2, 6), reduce=[[('skip_connect', 1), ('dil_conv_3x3', 0)], [('max_pool_3x3', 1), ('dil_conv_3x3', 0)], [('dil_conv_5x5', 3), ('max_pool_3x3', 0)], [('skip_connect', 3), ('dil_conv_3x3', 2)]], reduce_concat=range(2, 6))"
}


def to_dag(C_in, gene, reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](C_in, stride, True)
            if not isinstance(op, ops.Identity):
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s):
    return eval(s)


def parse(alpha, k):
    gene = []
    assert PRIMITIVES[-1] == 'none'

    for edges in alpha:
        edge_max, primitive_indices = torch.topk(edges[:, :-1], 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene


def parse_numpy(alpha, k):
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]
    each node has two edges (k=2) in CNN.
    """

    gene = []
    assert PRIMITIVES[-1] == 'none' # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        edge_max, primitive_indices = torch.topk(torch.tensor(edges[:, :-1]), 1) # ignore 'none'
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []
        for edge_idx in topk_edge_indices:
            prim_idx = primitive_indices[edge_idx]
            prim = PRIMITIVES[prim_idx]
            node_gene.append((prim, edge_idx.item()))

        gene.append(node_gene)

    return gene

