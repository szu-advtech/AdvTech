#Resnet18 training process and corresponding result are in 
https://www.kaggle.com/code/wasonlee/resnet18/edit/run/118534091
you must download the resnet18 model from the url,and put it in experiment/model

#Dissection process is in the experiment/
single_dissection.py:dissection for just one layer.
Sequential_dissection.py:dissection for sequential layers.

#Usage for Sequential_dissection.py
cd experiment
python Sequential_dissection.py --model [dissect_model] --dataset [dataset which the pre-training model trained in] --layer [stopping layer] --batch_size [] --quantile[] --dissect_units_perscent []
