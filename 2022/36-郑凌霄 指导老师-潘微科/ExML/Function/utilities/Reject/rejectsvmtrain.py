import numpy as np
import cvxopt
from .PSD import PSD

def quadprog(H, f, L=None, k=None, Aeq=None, beq=None, lb=None, ub=None):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    n_var = H.shape[1]

    P = cvxopt.matrix(H, tc='d')
    q = cvxopt.matrix(f, tc='d')

    if L is not None or k is not None:
        assert(k is not None and L is not None)
        if lb is not None:
            L = np.vstack([L, -np.eye(n_var)])
            k = np.vstack([k, -lb])

        if ub is not None:
            L = np.vstack([L, np.eye(n_var)])
            k = np.vstack([k, ub])

        L = cvxopt.matrix(L, tc='d')
        k = cvxopt.matrix(k, tc='d')

    if Aeq is not None or beq is not None:
        assert(Aeq is not None and beq is not None)
        Aeq = cvxopt.matrix(Aeq, tc='d')
        beq = cvxopt.matrix(beq, tc='d')

    sol = cvxopt.solvers.qp(P, q, L, k, Aeq, beq)

    return np.array(sol['x'])


def rejectsvmtrain(y_tr,X_tr,cost,C1,C2,opts):
#This function is used for training the rejection model

    labelSet = np.sort(np.unique(y_tr))
    y_temp = y_tr.copy()
    y_tr[y_temp == labelSet[0]] = 1.0
    try:
        y_tr[y_temp == labelSet[1]] = -1.0
    except:
        print('error')
    sample_num = np.shape(X_tr)[0]
    alpha = 1.0
    beta = 1.0/(1.0-2.0*cost)
    K = PSD(X_tr,opts)

    H = np.diag(np.append(np.append(C1/2*np.ones((1,sample_num)),C2/2*np.ones((1,sample_num))),np.zeros((1,sample_num))))
    H = (H + H.T)/2
    f = np.append(np.zeros((sample_num*2,1)),np.ones((sample_num,1)),axis=0)
    # the constrain b
    b1 = np.ones((sample_num,1))*(-cost)
    b2 = np.ones((sample_num,1))*(-1)
    b3 = np.zeros((sample_num,1))
    b = np.append(np.append(b1,b2,axis=0),b3,axis=0)
    # the constrain A:
    A1 = np.append(np.append(np.zeros((sample_num,sample_num)),-1*cost*beta*K,axis=1),-1*np.diag(np.ones((sample_num))),axis=1)
    A2 = np.append(np.append(-alpha/2*K*np.tile(y_tr,(1,sample_num)),alpha/2*K,axis=1),-1*np.diag(np.ones((sample_num))),axis=1)
    A3 = np.append(np.append(np.zeros((sample_num,sample_num)),np.zeros((sample_num,sample_num)),axis=1),-1*np.diag(np.ones((sample_num))),axis=1)
    A = np.append(np.append(A1,A2,axis=0),A3,axis=0)

    para = quadprog(H,f,None,None,A,b);
    model = {}
    model['w'] = para[0:sample_num]
    model['u'] = para[sample_num:2*sample_num]
    model['opts'] = opts
    model['X'] = X_tr

    return [model,labelSet]

