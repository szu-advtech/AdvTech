# If there is any problem in the following steps, please contact me, thank you very much

## Step1 Create a virtual environment

```sh
conda create -n ndc python=3.8
conda activate ndc
```

## Step2 Install the requirements

```sh
pip install -r requirements.txt
```

## Step3 Download the dataset and put it in the ./data 

Download Link : [groundtruth_UNDC.7z](https://pan.baidu.com/s/1upFvbaK2z8VPieaXEKfcKQ) (pwd: 1234)
unpack just like this:

```text
data
--groundtruth
  --gt_UNDC
```

## Step4 Train the Bool Network Frist

If you want to train, just use one of the following commands.
If you don't, just use the ./weight/net_bool-saved.pth to test.

> By the way, I set the max_epoch=3 which is enough, if you want to train more epoch just modify it in the ./config/trainer/default.yaml

```sh
# Train with gpu (recommend)
python src/train.py trainer=gpu model.train_float=False
# Train with cpu 
python src/train.py trainer=cpu model.train_float=False
```

After training the bool network, you can see all the result was saved in ./logs/train/runs/{the_time_you_train}.
In this folder, you can see the checkpoints and the objects sampled during the train.
You can also get some pth file saved in the ./weights.

## Step5 Train the Float Network

Choose any one pth file you like, use one of the following commands to train the float network.

```sh
# Train with gpu (recommend)
python src/train.py trainer=gpu model.train_float=True model.net_bool_pth="./weights/net_bool-saved.pth"
# Train with cpu 
python src/train.py trainer=cpu model.train_float=True model.net_bool_pth="./weights/net_bool-saved.pth"
# Train with gpu and other bool network weight file.
python src/train.py trainer=gpu model.train_float=True model.net_bool_pth="{?}"

```

After training the float network, you can see the final result of undc, which was saved in ./logs/train/runs/{the_time_you_train}.

## Step6 Eval the results
