#! /bin/bash
#python selflabel.py --gpu 0,1,2,3 --ct2 0.9  --num 0.05
#python selflabel.py --gpu 0,1,2,3 --ct2 0.9  --num 0.01

for i in 1;do
{
  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.9_0.02 --ct2 0.9  --num 0.02 &
  wait
  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.9_0.01 --ct2 0.9  --num 0.01 &
  wait
  python selflabel.py  --gpu 0,1,2,3  --version stl10_0.95_0.05 --ct2 0.95  --num 0.05 &
  wait
  python selflabel.py  --gpu 0,1,2,3  --version stl10_0.95_0.04 --ct2 0.95  --num 0.04 &
  wait
  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.95_0.03 --ct2 0.95  --num 0.03 &
  wait
  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.95_0.02 --ct2 0.95  --num 0.02 &
  wait
  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.95_0.01 --ct2 0.95  --num 0.01 &
  wait
  python selflabel.py  --gpu 0,1,2,3  --version stl10_0.98_0.05 --ct2 0.98  --num 0.05 &
  wait
  python selflabel.py  --gpu 0,1,2,3  --version stl10_0.98_0.04 --ct2 0.98  --num 0.04 &
  wait
  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.98_0.03 --ct2 0.98  --num 0.03 &
  wait
  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.98_0.02 --ct2 0.98  --num 0.02 &
  wait
  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.98_0.01 --ct2 0.98  --num 0.01 &
  wait
#  python selflabel.py  --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml --gpu 0,1,2,3  --version stl10_0.9_0.05 --ct2 0.9  --num 0.05 &
#  wait
#  python selflabel.py  --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml --gpu 0,1,2,3  --version stl10_0.9_0.04 --ct2 0.9  --num 0.04 &
#  wait
#  python selflabel.py  --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml --gpu 0,1,2,3 --version stl10_0.9_0.03 --ct2 0.9  --num 0.03 &
#  wait
#  python selflabel.py  --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml --gpu 0,1,2,3 --version stl10_0.9_0.02 --ct2 0.9  --num 0.02 &
#  wait
#  python selflabel.py  --config_env configs/env.yml --config_exp configs/selflabel/selflabel_cifar10.yml --gpu 0,1,2,3 --version stl10_0.9_0.01 --ct2 0.9  --num 0.01 &
#  wait
}
done