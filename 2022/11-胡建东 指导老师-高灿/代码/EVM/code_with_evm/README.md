# Requirements:
	- python > 3.6
	- libMR
	- pandas
	- numpy
	- sklearn
	- matplotlib
	- seaborn
	- hyperopt
	
	
# Installation:
```sh
$ pip install libmr, pandas, numpy, sklearn, matplotlib, seaborn, hyperopt
```	


# Usage:
	- 'sample_data_prep.py' contains code for modelling letter-recongition data to openset condition.
	- 'config.py' contains hyperparameter values related to EVM model.
	- 'EVM.py' contains code for Extreme Value Machine OpenSet algorithm. training and testing data is passed in csv format.
	- 'metrics.py' is used to obtain metrics such as Confusion Matrix, F-measure, Recognition Accuracy, Precision, Recall. 
	- 'Hyperparameter_tuning.py' performs hyperparameter tuning for your dataset using Hyperopt library.
	
	
# Attribution:

This is an implementation of the Extreme Value Machine by Rudd et al., with minor changes from the original work.


@article{rudd2018extreme, title={The extreme value machine}, author={Rudd, Ethan M and Jain, Lalit P and Scheirer, Walter J and Boult, Terrance E}, journal={IEEE transactions on pattern analysis and machine intelligence}, volume={40}, number={3}, pages={762--768}, year={2018}, publisher={IEEE} }