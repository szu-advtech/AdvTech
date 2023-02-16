# convert IPv6 str to numpy seeds.npy

import numpy as np
from IPy import IP


with open("./seeds") as f:
    arrs = []
    for ip in f.read().splitlines()[:10000]:
        arrs.append([int(x, 16)
                    for x in IP(ip).strFullsize().replace(":", "")])

    np.save("seeds.npy", np.array(arrs, dtype=np.uint8))

