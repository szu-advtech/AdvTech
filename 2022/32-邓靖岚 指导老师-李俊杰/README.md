# Truncated variational GMM

This is the variational GMM algorithm of the paper:  
[*Can clustering scale sublinearly with its clusters? A variational EM acceleration of GMMs and k-means*](http://proceedings.mlr.press/v84/forster18a)  
Dennis Forster and Jörg Lücke, AISTATS, 2018

## 1. Prerequisites
- Python versions: 2.7.x or 3.x
- Python packages:
  - numpy
  - scipy
  - sklearn
  - matplotlib
  - mpi4py
  - kmc2

## 2. Execution

For execution, run  
`python main.py [algorithm] [parameters]`  
where `[algorithm]` is either `var-GMM-X`, `var-GMM-X+1`, `var-GMM-S` or `var-GMM-S+1`,  
and `[parameters]` is an optional list of free model parameters.

To start, you can for example simply run
`python main.py var-GMM-S+1`

## 3. Parameters  
- `dataset=BIRCH[dims]-[clusters]-[nsamples]`, default: `dataset=BIRCH2-400-100`. Parameters for artificial BIRCH data set. `[dims]` is the dimensionality and `[clusters]` is the number of BIRCH clusters and `[nsamples]` is the number of data points per clusters.    
- `C=[int]`, default: `C=400`. Number of clusters.  
- `Cprime=[int]`, default: `Cprime=5`. Truncation paramer: number of non-zero values in posteriors. `Cprime=1` recovers k-means-like algorithms.
- `G=[int]`, default: `G=5`. Cluster neighborhood size. Number of clusters that are considered as neighbors to each cluster c (including c itself) in the variational algorithms.
- `Niter=[int]`, default: `Niter=25`. Number of learning iterations.  
- `Ninit=[int]`, default: `Ninit=5`. Number of initial E-step iterations to find better initial neighbor guesses.  
- `VERBOSE=[True/False]`, default: `VERBOSE=True`. If `True` the loglikelihood, free energy, quantization error, purity, NMI, AMI and number of distance evaluations in each iteration are calculated and saved in the `/output/` directory, as well as an image of the current clusters and the data set. If `False`, only the number of distance evaluations are counted and saved per iteration. After learning is complete, all final values will be calculated and saved.

## 4. Disclaimer

This algorithm is not optimized for fast execution speed. As measure for speed, we show the number of distance evaluations wrt the number of distance evaluations used by full GMMs/k-means per iteration.   

For an implementation of the var-GMM-S algorithm with focus on fast real-time execution speed, see the following repository which uses C++ source code [vc-GMM](https://github.com/variational-sublinear-clustering/vc-GMM). The repository is part of the github-organization [Variational Sublinear Clustering](https://github.com/variational-sublinear-clustering), which will integrate further developments/repositories.
