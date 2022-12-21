import utils as utils
import bnlearn as bn
class Pair:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __eq__(self, other):
        return self.a == other.a and self.b == other.b
    def __hash__(self):
        return hash((self.a, self.b))
def entropic_peeling(df):
    I = set() # set of pairs found to be conditionally independent
    T = []# list of nodes in topological order
    FR = frozenset(df.columns) # all nodes, immutable
    nodes_count = df.shape[1]
    R = set(df.columns) # set of remaining nodes
    dict = {}
    for i in range(nodes_count):
        dict[df.columns[i]] = i
    while (len(R)>0):
        N = set() # set of nodes discovered as non-sources
        C = FR.difference(R) # condition on previous sources
        for Xi in R: # iterate rest nodes
            for Xj in R:
                if Xi == Xj: continue
                if Xi not in N and Xj not in N and Pair(Xi, Xj) not in I:
                    if utils.CHI_SQUARE(Xi, Xj, list(C), df.copy(True)):
                        I.update({Pair(Xi, Xj), Pair(Xj, Xi)})
                    else:
                        if utils.Oracle_Jinke(Xi, Xj, df.copy(True)):
                            N.add(Xj)
                        else:
                            N.add(Xi)
        S = R.difference(N) # we get some new source nodes
        R = R.difference(S) # update the remaining nodes
        for X in S:
            T.append(X)
    ansSoruce = []
    ansTarget = []
    # auxiliary
    skeleton, O = utils.getSkeleton(df)
    iSet = set()
    for i in range(0, len(T)):
        jSet = set(iSet) # T[0:i-1]
        for j in range(i+1, len(T)):
            if utils.CHI_SQUARE(T[i], T[j], list(jSet), df.copy(True)) == False and skeleton[dict[T[i]]][dict[T[j]]] == True:
                ansSoruce.append(T[i])
                ansTarget.append(T[j])
            jSet.add(T[j])# jSet = {T[0:j-1]} \ {T[i]}
        iSet.add(T[i])
    edges = {'source': ansSoruce, 'target': ansTarget}
    return utils.getAdjmat(edges, df)
def debug():
    model = bn.import_DAG("sachs")
    iterations = 10
    shd_sum = 0
    for i in range(0, iterations):
        df = bn.sampling(model, n=10000)
        correct_adjmat = model['adjmat']
        peeling_adjmat = entropic_peeling(df)
        shd = utils.SHD(correct_adjmat, peeling_adjmat)
        print(peeling_adjmat)
        print("第%d次：" %(i+1))
        print("peeling shd : %d" %shd)
        shd_sum += shd
    print("avg shd %f" %(shd_sum/iterations))
# debug()