import sys
from graphviz import Digraph
import genotypes as gt


def plot(genotype, file_path, caption=None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor='lightblue')

    for i, edges in enumerate(genotype):
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)

    
def convert_genotype_to_sample(geno):
    """
    geno is a genotypes.Genotype
    output the samples of normal and reduce cell
    """
    
    normal = geno.normal # list
    reduce = geno.reduce # list
    
    samples = []
    
    for cell in [normal, reduce]:
        sample = [[7, 7], [7, 7, 7], [7, 7, 7, 7], [7, 7, 7, 7, 7]]  # all ops are initialized to none
        for i, edges in enumerate(cell):
            for op, j in edges:
                try:
                    sample[i][j] = gt.PRIMITIVES.index(op)
                except:
                    raise ValueError('op {} can not be parsed'.format(op))
        result = []
        for node_s in sample:
            result.extend(node_s)
            
        samples.append(result)
    print('sample for normal:', samples[0])
    print('sample for reduce:', samples[1])
    print('sample for genotype:', str(samples[0] + samples[1]))

    
                
if __name__ == '__main__':
    print("")
    genotype_str = sys.argv[1]
    
    try:
        genotype = gt.from_str(genotype_str)
        convert_genotype_to_sample(genotype)
        
    except AttributeError:
        raise ValueError("Cannot parse {}".format(genotype_str))

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")
