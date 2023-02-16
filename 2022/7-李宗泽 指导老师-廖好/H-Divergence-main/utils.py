import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

def JSV_Gaussian_u(Fea, len_s, n_components, vtype):
    """compute one v-div with GMM value"""
    X = Fea[0:len_s, :] # fetch the sample 1
    Y = Fea[len_s:, :] # fetch the sample 2
    ind = np.random.choice(Fea.shape[0], len_s, replace=False)
    XY = Fea[ind]
    model_x = GaussianMixture(n_components=n_components)
    model_y = GaussianMixture(n_components=n_components)
    model_xy = GaussianMixture(n_components=n_components)
    model_x.fit(X)
    model_y.fit(Y)
    model_xy.fit(XY)

    if vtype == "vjs":
        mixed = 1/2*np.mean(-model_xy.score_samples(X))+1/2*np.mean(-model_xy.score_samples(Y))
        x_prob = 1/2*np.mean(-model_x.score_samples(X))
        y_prob = 1/2*np.mean(-model_y.score_samples(Y))

        jsv = mixed - x_prob - y_prob
    elif vtype == "vmin":
        mixed = 1/2*np.mean(-model_xy.score_samples(X))+1/2*np.mean(-model_xy.score_samples(Y))
        # mixed = np.mean(-model_xy.score_samples(XY))
        x_prob = np.mean(-model_x.score_samples(X))
        y_prob = np.mean(-model_y.score_samples(Y))
        jsv = mixed - min(x_prob, y_prob)
    else:
        raise NotImplementedError("Please implement your desired v-divergence")
    return jsv

def JSV_Gaussian(Fea, N_per, N1, alpha, n_components, dtype, vtype):
    """run permutation test with v-div with GMM"""
    jsv_vector = np.zeros(N_per)
    jsv_value = JSV_Gaussian_u(Fea, N1, n_components, vtype)
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Fea[indx]
        # print(Kx)
        Ky = Fea[indy]
        indxy = np.concatenate((indx[:int(nx/2)], indy[:int(nx/2)]))
        Kxy = Fea[indxy]

        model_x = GaussianMixture(n_components=n_components)
        model_y = GaussianMixture(n_components=n_components)
        model_xy = GaussianMixture(n_components=n_components)
        model_x.fit(Kx)
        model_y.fit(Ky)
        model_xy.fit(Kxy)

        if vtype == "vjs":
            mixed = 1/2*np.mean(-model_xy.score_samples(Kx))+1/2*np.mean(-model_xy.score_samples(Ky))
            x_prob = 1/2*np.mean(-model_x.score_samples(Kx))
            y_prob = 1/2*np.mean(-model_y.score_samples(Ky))

            jsv_r = mixed - x_prob - y_prob
        elif vtype == "vmin":
            mixed = 1/2*np.mean(-model_xy.score_samples(Kx))+1/2*np.mean(-model_xy.score_samples(Ky))
            x_prob = np.mean(-model_x.score_samples(Kx))
            y_prob = np.mean(-model_y.score_samples(Ky))

            jsv_r = mixed - min(x_prob, y_prob)
        else:
            raise NotImplementedError("Please implement your desired v-divergence")

        jsv_vector[r] = jsv_r

        if jsv_vector[r] > jsv_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_jsv_vector = np.sort(jsv_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_jsv_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, jsv_value

def JSV_KDE_u(Fea, len_s, bandwidth, vtype):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s] # fetch the sample 1
    Y = Fea[len_s:] # fetch the sample 2
    ind = np.random.choice(Fea.shape[0], len_s, replace=False)
    XY = Fea[ind]
    model_x = KernelDensity(bandwidth=bandwidth[0])
    model_y = KernelDensity(bandwidth=bandwidth[1])
    model_xy = KernelDensity(bandwidth=bandwidth[2])
    model_x.fit(X)
    model_y.fit(Y)
    model_xy.fit(XY)

    if vtype == "vjs":
        mixed = 1/2*np.mean(-model_xy.score_samples(X))+1/2*np.mean(-model_xy.score_samples(Y))
        x_prob = 1/2*np.mean(-model_x.score_samples(X))
        y_prob = 1/2*np.mean(-model_y.score_samples(Y))

        jsv = mixed - x_prob - y_prob
    elif vtype == "vmin":
        mixed = 1/2*np.mean(-model_xy.score_samples(X))+1/2*np.mean(-model_xy.score_samples(Y))
        x_prob = np.mean(-model_x.score_samples(X))
        y_prob = np.mean(-model_y.score_samples(Y))

        jsv = mixed - min(x_prob, y_prob)
    else:
        raise NotImplementedError("Please implement your desired v-divergence")
    return jsv

def JSV_KDE(Fea, N_per, N1, alpha, bandwidth, dtype, vtype):
    """run two-sample test (TST) using ordinary Gaussian kernel."""
    jsv_vector = np.zeros(N_per)
    jsv_value = JSV_KDE_u(Fea, N1, bandwidth, vtype)
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Fea[indx]
        # print(Kx)
        Ky = Fea[indy]
        indxy = np.concatenate((indx[:int(nx/2)], indy[:int(nx/2)]))
        Kxy = Fea[indxy]

        model_x = KernelDensity(bandwidth=bandwidth[0])
        model_y = KernelDensity(bandwidth=bandwidth[1])
        model_xy = KernelDensity(bandwidth=bandwidth[2])
        model_x.fit(Kx)
        model_y.fit(Ky)
        model_xy.fit(Kxy)

        if vtype == "vjs":
            mixed = 1/2*np.mean(-model_xy.score_samples(Kx))+1/2*np.mean(-model_xy.score_samples(Ky))
            x_prob = 1/2*np.mean(-model_x.score_samples(Kx))
            y_prob = 1/2*np.mean(-model_y.score_samples(Ky))

            jsv_r = mixed - x_prob - y_prob
        elif vtype == "vmin":
            mixed = 1/2*np.mean(-model_xy.score_samples(Kx))+1/2*np.mean(-model_xy.score_samples(Ky))
            x_prob = np.mean(-model_x.score_samples(Kx))
            y_prob = np.mean(-model_y.score_samples(Ky))

            jsv_r = mixed - min(x_prob, y_prob)
        else:
            raise NotImplementedError("Please implement your desired v-divergence")

        jsv_vector[r] = jsv_r

    S_jsv_vector = np.sort(jsv_vector)
    threshold = S_jsv_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if jsv_value > threshold:
        h = 1
    return h, threshold, jsv_value
