from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
NODE_INDEX = (
    'identity',
    'max_pool_3x3',
    'avg_pool_3x3',
    'squeeze_and_excitation',
    'lb_conv_3x3',
    'lb_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'conv_7x1_1x7',
)


def decode(genome):

    assert len(genome) % 8 == 0
    normal_cell = genome[0:len(genome)//2]
    reduce_cell = genome[len(genome)//2:]
    normal = []
    reduce = []
    normal_concat = list(range(2, len(normal_cell)//4 + 2))
    reduce_concat = list(range(2, len(reduce_cell)//4 + 2))

    for i in range(len(normal_cell)//2):
        normal.append((NODE_INDEX[normal_cell[2*i]], normal_cell[2*i+1]))
        if normal_cell[2*i + 1] in normal_concat:
            normal_concat.remove(normal_cell[2 * i + 1])

        reduce.append((NODE_INDEX[reduce_cell[2*i]], reduce_cell[2*i+1]))
        if reduce_cell[2*i + 1] in reduce_concat:
            reduce_concat.remove(reduce_cell[2 * i + 1])

    genotype = Genotype(
        normal=normal,
        normal_concat=normal_concat,
        reduce=reduce,
        reduce_concat=reduce_concat
    )
    return genotype


TestNet = Genotype(normal=[('sep_conv_7x7', 0),
                           ('dil_conv_5x5', 1),
                           ('sep_conv_5x5', 0),
                           ('sep_conv_5x5', 0),
                           ('lb_conv_3x3', 2),
                           ('max_pool_3x3', 2),
                           ('sep_conv_5x5', 0),
                           ('lb_conv_3x3', 4),
                           ('dil_conv_3x3', 2),
                           ('lb_conv_5x5', 2)],
                   normal_concat=[3, 5, 6],
                   reduce=[('squeeze_and_excitation', 1),
                           ('sep_conv_7x7', 1),
                           ('sep_conv_5x5', 2),
                           ('identity', 1),
                           ('sep_conv_5x5', 3),
                           ('max_pool_3x3', 3),
                           ('avg_pool_3x3', 0),
                           ('dil_conv_3x3', 0),
                           ('lb_conv_3x3', 5),
                           ('squeeze_and_excitation', 1)],
                   reduce_concat=[4, 6])

#             [OP1, CONN1, OP2, CONN2]
test_genome = [0, 0, 1, 1,      # node 1    Normal
               2, 1, 3, 2,      # node 2
               4, 2, 5, 3,      # node 3
               6, 3, 7, 2,      # node 4
               8, 2, 9, 0,      # node 5
               0, 0, 1, 1,      # node 1    Reduction
               2, 1, 3, 2,      # node 2
               4, 2, 5, 3,      # node 3
               6, 3, 7, 2,      # node 4
               8, 2, 9, 0, ]    # node 5

# if __name__ == '__main__':
#
#     # test_genotype = decode(test_genome)
#     # print(test_genotype)

