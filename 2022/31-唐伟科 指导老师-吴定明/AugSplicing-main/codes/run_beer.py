from codes.invoke import optimiAlgo
import time

if __name__ == '__main__':
    start_time = int(round(time.time() * 1000))
    'BeerAdvocate data'
    inputfile = '../data/BeerAdvocate/input.tensor'
    outpath = '../output/BeerAdvocate/AugSplicing'
    s = 15  # time stride(day)
    maxSp = 30  # the maximum splicing number at each epoch
    delimeter, N = ',', 3  # the delimeter/dimension of input data
    steps = 30  # the number of time steps
    k, l = 10, 5  # the number of top blocks we find/ slack constant
    optimiAlgo(inputfile, outpath, s, k, l, maxSp, N, delimeter, steps=30)
    end_time = int(round(time.time() * 1000))
    print("消耗时间为" + str(int(end_time - start_time)) + "毫秒")
