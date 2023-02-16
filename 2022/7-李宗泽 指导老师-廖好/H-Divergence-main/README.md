# H-Divergence

***Notice: This repo is still under construction.***

### HIGGS

Get the HIGGS data from here - https://drive.google.com/file/d/1sHIIFCoHbauk6Mkb6e8a_tp1qnvuUOCc/view

To run HIGGS experiment, choose --n = 500,1000,1500,2500,4000,5000 and run
```
python higgs.py --n 500
```


### MNIST

Get the fake MNIST data from here - https://drive.google.com/file/d/13JpGbp7PEm4PfZ6VeqpFiy0lHfVpy5Z5/view

To run MNIST experiment, choose --n = 100,200,300,400,500 and run
```
python mnist.py --n 100
```

### Blob

To run Blob experiment with KDE
```
python blob_kde.py --exptype power --vtype vmin
```
To run Blob experiment with GMM
```
python blob_gmm.py --exptype power --vtype vmin
```

### HDGM

To run HDGM experiment, choose --vtype = vjs, --n = 100,1000,1500,2500, and --d = 3,5,10,15,20 and run
```
python hdgm.py --exptype power --vtype vjs --n 100 --d 3
```
