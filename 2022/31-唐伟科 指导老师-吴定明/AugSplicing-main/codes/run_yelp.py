from codes.invoke import optimiAlgo


if __name__ == '__main__':
    'yelp data'
    inputfile = '../data/yelp/input.tensor'
    outpath = '../output/yelp/AugSplicing'
    s = 30  # time stride(day)
    maxSp = 30  # the maximum splicing number at each epoch
    delimeter, N = ',', 3  # the delimeter/dimension of input data
    steps = 30  # the number of time steps
    k, l = 10, 5  # the number of top blocks we find/ slack constant
    optimiAlgo(inputfile, outpath, s, k, l, maxSp, N, delimeter, steps=30)
