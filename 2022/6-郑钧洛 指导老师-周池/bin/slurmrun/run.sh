srun -N 3 -n 3  ../runner
srun -N 3 -n 3 ../linear ../../setup/MNIST
srun -N 3 -n 3 ../logistic ../../setup/MNIST
srun -N 3 -n 3  ../MLP ../../setup/MNIST
srun -N 3 -n 3 ../conv ../../setup/MNIST
srun -N 3 -n 3  ../RNN
srun -N 3 -n 3 ../SVM ../../setup/MNIST