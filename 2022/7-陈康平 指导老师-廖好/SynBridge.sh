#!/usr/bin/env bash

seeds=(123 321 111)
splits=(1 2 3)
for seed in ${seeds[@]};
do
    for split in ${splits[@]};
    do
        # =============================================================== R>L =====================================================================
        python train_bridge.py  --source restaurant --target laptop --split $split --use_unlabel 1 --use_syntactic 1 --use_semantic 0 --name [SynBridge] --seed $seed
        # =============================================================== L>R =====================================================================
        python train_bridge.py  --source laptop --target restaurant --split $split --use_unlabel 1 --use_syntactic 1 --use_semantic 0 --name [SynBridge] --seed $seed
        # =============================================================== R>D =====================================================================
        python train_bridge.py  --source restaurant --target device --split $split --use_unlabel 1 --use_syntactic 1 --use_semantic 0 --name [SynBridge] --seed $seed
        # =============================================================== D>R =====================================================================
        python train_bridge.py  --source device --target restaurant --split $split --use_unlabel 1 --use_syntactic 1 --use_semantic 0 --name [SynBridge] --seed $seed
        # =============================================================== L>D =====================================================================
        python train_bridge.py  --source laptop --target device --split $split --use_unlabel 1 --use_syntactic 1 --use_semantic 0 --name [SynBridge] --seed $seed        #
        # =============================================================== D>L =====================================================================
        python train_bridge.py  --source device --target laptop --split $split --use_unlabel 1 --use_syntactic 1 --use_semantic 0 --name [SynBridge] --seed $seed
    done
done