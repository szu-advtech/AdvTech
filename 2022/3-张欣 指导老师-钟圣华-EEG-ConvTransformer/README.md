# EEG-ConvTransformer

This is a community implemention of arcticle:

`[1] Bagchi S, Bathula D R. EEG-ConvTransformer for single-trial EEG-based visual stimulus classification[J]. Pattern Recognition, 2022, 129: 108757.`

The proposed method (Called EEG-ConvTransformer) of citation[1] is implemented in /model. It should be no problem.

## Usage
Before training, you need:

1) Download the dataset from https://purl.stanford.edu/bq914sc3730
2) Download the electrodes XYZ locations from: ftp://ftp.egi.com/pub/support/Documents/net_layouts/hcgsn_128.pdf 
Note that I can not download this PDF, if you get it please send me a copy, thanks. 
I used a 64-channels version from https://github.com/numediart/EEGLearn-Pytorch/blob/master/Sample%20Data/Neuroscan_locs_orig.mat
It depends on what dataset you use.
3) Perform pre-processing by run data_load/serialize.py. This script will do `Azimuthal Equidistant Projection(AEP)[2]` and pkl serialization in Parallel. 
The EEG raw data will be transformed to images[1][2].
Remember to pass two filepath.

Now, You can run main.py to start you training task. Testing or Validation can be easily writen by modify this file.

## Problems
1) Uncertain coding when do AEP in /data_load/serialize.py due to undisclosed details in citation [2]: 

&nbsp; &nbsp; `"However, contrary to the three frequency power bands from the earlier work, the AEP and interpolation are applied to the preprocessed signal to form a single channel mesh of G1 × G2 per time-frame."`. 

&nbsp; &nbsp; Read chapter 3.1 of citation[1] for more details. It's welcome to help me to refine this repository.

2) I got time-out error when download 128-channels version electrodes XYZ locations from: ftp://ftp.egi.com/pub/support/Documents/net_layouts/hcgsn_128.pdf.
I have to down-sample EEG raw data in channels dim to make it compatible with 64-channels version.

### Refs

[1] Bagchi S, Bathula D R. EEG-ConvTransformer for single-trial EEG-based visual stimulus classification[J]. Pattern Recognition, 2022, 129: 108757.

[2] Bashivan, et al. "Learning Representations from EEG with Deep Recurrent-Convolutional Neural Networks." International conference on learning representations (2016).

[3] https://github.com/pbashivan/EEGLearn

[4] https://github.com/numediart/EEGLearn-Pytorch

## Note
Here is a bound version and will not be updated, You may want to find the author edition:
https://github.com/MeetXinZhang/EEG-ConvTransformer
