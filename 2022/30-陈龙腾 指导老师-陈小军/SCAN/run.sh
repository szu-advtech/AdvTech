#! /bin/bash
#python selflabel.py --gpu 0,1,2,3 --ct2 0.9  --num 0.05
#python selflabel.py --gpu 0,1,2,3 --ct2 0.9  --num 0.01

for i in 1;do
{
#  python selflabel.py  --gpu 0,1,2,3 --epochs 50 --topk_lp 5 --version stl10_0.95_0.03_2k5 --ct2 0.95  --num 0.03 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3 --epochs 50 --topk_lp 25 --version stl10_0.95_0.03_2k25 --ct2 0.95  --num 0.03 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3 --epochs 100 --version stl10_0.95_0.03_epo100 --ct2 0.95  --num 0.03 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3 --epochs 150 --version stl10_0.95_0.03_epo150 --ct2 0.95  --num 0.03 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3  --epoch 50 --version stl10_0.9_0.05 --ct2 0.9  --num 0.05 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3  --epoch 50 --version stl10_0.9_0.04 --ct2 0.9  --num 0.04 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.9_0.03 --ct2 0.9  --num 0.03 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.9_0.02 --ct2 0.9  --num 0.02 &
#  wait
  python selflabel.py  --epochs 50 --gpu 0,1,2,3 --version stl10_0.9_0.01_v2 --ct2 0.9  --num 0.01 &
  wait
#  python selflabel.py  --epochs 100 --gpu 0,1,2,3 --version stl10_0.9_0.01_ep100 --ct2 0.9  --num 0.01 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.85_0.03 --ct2 0.85  --num 0.03 &
#  wait
#  python selflabel.py  --gpu 0,1,2,3 --version stl10_0.8_0.03 --ct2 0.8  --num 0.03 &
#  wait
}
done