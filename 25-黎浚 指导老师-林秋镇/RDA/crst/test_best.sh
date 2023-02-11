
for i in {1800..6001..200}
do
   echo "TEST $i MODEL"
   python evaluate_advent_best.py --test-flipping --data-dir /root/autodl-tmp/Cityscapes --restore-from /root/autodl-nas/model_$i.pth --save ../RDA/experiments/GTA2Cityscapes_RDA
done
