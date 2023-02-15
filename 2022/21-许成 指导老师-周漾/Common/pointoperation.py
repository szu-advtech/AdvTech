import numpy as np

def normalize_point_cloud(inputs):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    #print("shape",input.shape)
    C = inputs.shape[-1]
    pc = inputs[:,:,:3]
    if C > 3:
        nor = inputs[:,:,3:]

    centroid = np.mean(pc, axis=1, keepdims=True)
    pc = inputs[:,:,:3] - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(pc ** 2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    pc = pc / furthest_distance
    if C > 3:
        return np.concatenate([pc,nor],axis=-1)
    else:
        return pc

def normalize_simple_point_cloud(input):
    """
    input: pc [P,3]
    output: pc , centroid ,furthest_distance
    """
    pc = input[:,:3]
    centroid = np.mean(pc,axis=0,keepdims=True)
    pc = input[:,:3]-centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(pc**2,axis=-1,keepdims=True)),axis=0,keepdims=True
    )
    pc = pc/furthest_distance
    return pc