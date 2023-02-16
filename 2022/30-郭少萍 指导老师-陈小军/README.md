# 1. How to install

This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt`  



# 2. Datasets

Follow [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to install the datasets.



# 3. How to Run

The running scripts are provided in `scripts/`, which allow you to reproduce the results .

Make sure you change the path in `DATA` and run the commands under the project's  current directory .

`DATASET` takes as input a dataset name, like `oxford_pets` or `caltech101`. The valid names are the files names in `./configs/datasets/`.



## 1) ZeroshotCLIP

You will need both `scripts/zsclip/base2new_train.sh` and `scripts/zsclip/base2new_test.sh`. The former evaluates a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have one input arguments, i.e., `DATASET` 

Below we provide an example on how to evaluate the model on oxford_pets.

To get the performance on the base classes, run

```bash
bash scripts/zsclip/base2new_train.sh oxford_pets
```

To get the  performance on the new classes, run

```bash
bash scripts/zsclip/base2new_test.sh oxford_pets
```

For instance, after you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on oxford_pets using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– oxford_pets/
|   |   |   |–– shots_16/
|   |   |   |   |–– ZeroshotCLIP/
|   |   |   |   |   |–– vit_b16/
|   |   |   |   |   |   |–– log.txt

|   |–– train_base/
|   |   |–– oxford_pets/
|   |   |   |–– shots_16/
|   |   |   |   |–– ZeroshotCLIP/
|   |   |   |   |   |–– vit_b16/
|   |   |   |   |   |   |–– log.txt


```



## 2) CoOp

You will need both `scripts/coop/base2new_train.sh` and `scripts/coop/base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

Below we provide an example on how to evaluate the model on oxford_pets.

```bash
# seed=1
bash scripts/coop/base2new_train.sh oxford_pets 1
bash scripts/coop/base2new_test.sh oxford_pets 1

# seed=2
bash scripts/coop/base2new_train.sh oxford_pets 2
bash scripts/coop/base2new_test.sh oxford_pets 2

# seed=3
bash scripts/coop/base2new_train.sh oxford_pets 3
bash scripts/coop/base2new_test.sh oxford_pets 3
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/oxford_pets/shots_16/CoOp/vit_b16_ctxv1
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/oxford_pets/shots_16/CoOp/vit_b16_ctxv1 --test-log
```



## 3) CoCoOP

You will need both `scripts/cocoop/base2new_train.sh` and `scripts/cocoop/base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

Below we provide an example on how to evaluate the model on oxford_pets.

```bash
# seed=1
bash scripts/cocoop/base2new_train.sh oxford_pets 1
bash scripts/cocoop/base2new_test.sh oxford_pets 1

# seed=2
bash scripts/cocoop/base2new_train.sh oxford_pets 2
bash scripts/cocoop/base2new_test.sh oxford_pets 2

# seed=3
bash scripts/cocoop/base2new_train.sh oxford_pets 3
bash scripts/cocoop/base2new_test.sh oxford_pets 3

```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results. For instance, after you finish the evaluation (including `base2new_train.sh` and `base2new_test.sh`) on oxford_pets using the aforementioned commands, you would get

```
output
|–– base2new/
|   |–– test_new/
|   |   |–– oxford_pets/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
|   |–– train_base/
|   |   |–– oxford_pets/
|   |   |   |–– shots_16/
|   |   |   |   |–– CoCoOp/
|   |   |   |   |   |–– vit_b16_c4_ep10_batch1_ctxv1/
|   |   |   |   |   |   |–– seed1/
|   |   |   |   |   |   |–– seed2/
|   |   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/oxford_pets/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/oxford_pets/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1 --test-log
```



## 4) CAPG

You will need both `scripts/capg/base2new_train.sh` and `scripts/capg/base2new_test.sh`. The former trains a model on bash classes while the latter evaluates the trained model on new classes. Both scripts have two input arguments, i.e., `DATASET` and `SEED`.

Below we provide an example on how to evaluate the model on oxford_pets.

```bash
# seed=1
bash scripts/capg/base2new_train.sh oxford_pets 1
bash scripts/capg/base2new_test.sh oxford_pets 1

# seed=2
bash scripts/capg/base2new_train.sh oxford_pets 2
bash scripts/capg/base2new_test.sh oxford_pets 2

# seed=3
bash scripts/capg/base2new_train.sh oxford_pets 3
bash scripts/capg/base2new_test.sh oxford_pets 3

```

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/oxford_pets/shots_16/CAPG/vit_b16_c4_ep10_batch1
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/oxford_pets/shots_16/CAPG/vit_b16_c4_ep10_batch1 --test-log
```



