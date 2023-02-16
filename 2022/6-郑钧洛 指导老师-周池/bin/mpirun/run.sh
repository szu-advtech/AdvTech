read -p $'Please enter a number to choose dataset:1.VGGFace2  2.NIST  3.SYNTHETIC  4.MNIST \x0a' dset 
if [ $dset = 1 ]
then
    mpirun -np 3  ../runner 
    mpirun -np 3  ../linear ../../setup/VGG
    mpirun -np 3 ../logistic ../../setup/VGG
    mpirun -np 3  ../MLP ../../setup/VGG
    mpirun -np 3 ../conv ../../setup/VGG
elif [ $dset = 2 ]
then
    mpirun -np 3  ../runner 
    mpirun -np 3  ../linear ../../setup/NIST
    mpirun -np 3 ../logistic ../../setup/NIST
    mpirun -np 3  ../MLP ../../setup/NIST
    mpirun -np 3 ../conv ../../setup/NIST
elif [ $dset = 3 ]
then
    mpirun -np 3  ../runner 
    mpirun -np 3  ../linear ../../setup/SYN
    mpirun -np 3 ../logistic ../../setup/SYN
    mpirun -np 3  ../MLP ../../setup/SYN
    mpirun -np 3 ../conv ../../setup/SYN
elif [ $dset = 4 ]
then
    mpirun -np 3  ../runner 
    mpirun -np 3  ../linear ../../setup/MNIST
    mpirun -np 3 ../logistic ../../setup/MNIST
    mpirun -np 3  ../MLP ../../setup/MNIST
    mpirun -np 3 ../conv ../../setup/MNIST
fi