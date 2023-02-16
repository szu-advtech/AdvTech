# FLOP-covid


## Evironment

Python==3.6 + Pytorch==1.6.0

## Get datasets

1. Download the CovidX dataset following the instructions in https://github.com/lindawangg/COVID-Net. Create a file directory called datax and put the data into it.
2. Download the Kvaisr dataset (version 2) following the instructions in https://datasets.simula.no/kvasir/#data-collection.


## Command

### Covidx dataset under non-iid settings

Baseline

  python  federated_main_covid.py --model=CovidNet --dataset=covid --gpu=0 --iid=0 --num_local=0  --epochs=100  --seed=123 --local_ep=3  --exp_id=20210101001 --    num_users=5 



FLOP

  python  federated_main_covid.py --model=CovidNet --dataset=covid --gpu=0 --iid=0 --num_local=1  --epochs=100  --seed=123 --local_ep=3  --exp_id=20210101001 --    num_users=5 
  
  
### Kvasir dataset under non-iid settings

Baseline

  python   federated_main_kvasir.py --model=Mobile_Net --dataset=kvasir --gpu=0 --iid=0 --epochs=100  --seed=123 --local_ep=3  --exp_id=20210101001 --num_local=0 ----num_users=5 



FLOP

  python   federated_main_kvasir.py --model=Mobile_Net --dataset=kvasir --gpu=0 --iid=0 --epochs=100  --seed=123 --local_ep=3  --exp_id=20210101001 --num_local=1 ----num_users=5 
