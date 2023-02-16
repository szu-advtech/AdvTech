#!/usr/bin/env bash

#################################################
#### Dataset: MetaSense_Activity_Recognition ####
#################################################

dataset="metasense_activity_scaled"
model="MetaSense_Activity_model"
meta_lr="0.1"

#===========================
# Method: 'Src'
method="Src"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test  --src rest --train 1 --model $model
done

#===========================
# Method: 'Tgt'

method="Tgt"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 1 2 5 10
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot
  done
done

#===========================
# Method: 'Src_Tgt'


method="Src_Tgt"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 1 2 5 10
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot
  done
done

##===========================
## Method: 'TrC'


method="TrC"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 1 2 5 10
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --load_checkpoint_path log/metasense_activity_scaled/Src/src_rest/tgt_${tgt}/run_test/cp/
  done
done


##===========================
## Method: 'PN'


method="PN"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 5
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
  done
done

##===========================
## Method: 'MAML'


method="MAML"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 5
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
  done
done

##===========================
## Method: 'MetaSense'

method="MetaSense"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 5
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
  done
done


##################################################
###### Dataset: MetaSense_Speech_Recognition #####
##################################################


dataset="metasense_speech_scaled"
model="MetaSense_Speech_model"
meta_lr="0.15"

#===========================
# Method: 'Src'
method="Src"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test  --src rest --train 1 --model $model
done

#===========================
# Method: 'Tgt'

method="Tgt"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 1 2 5 10
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot
  done
done

#===========================
# Method: 'Src_Tgt'


method="Src_Tgt"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 1 2 5 10
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot
  done
done

##===========================
## Method: 'TrC'


method="TrC"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 1 2 5 10
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --load_checkpoint_path log/metasense_speech_scaled/Src/src_rest/tgt_${tgt}/run_test/cp/
  done
done


##===========================
## Method: 'PN'


method="PN"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 5
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
  done
done

##===========================
## Method: 'MAML'


method="MAML"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 5
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
  done
done

##===========================
## Method: 'MetaSense'

method="MetaSense"
for tgt in PH0007-jskim PH0012-thanh PH0014-wjlee PH0034-ykha PH0038-iygoo PH0041-hmkim PH0045-sjlee WA0002-bkkim WA0003-hskim WA4697-jhryu
do
  for nshot in 5
  do
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
  done
done




##################################################
################## Dataset: HHAR ##################
##################################################


dataset="hhar_scaled"
model="HHAR_model"
meta_lr="0.1"



#===========================
# Method: 'Src'

method="Src"
for d in nexus4 s3 s3mini lgwatch
do
  for p in a b c d e f
  do
    tgt=${d}..${p}
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test  --src rest --train 1 --model $model
  done
done

#===========================
# Method: 'Tgt'

method="Tgt"
for d in nexus4 s3 s3mini lgwatch
do
  for p in a b c d e f
  do
    tgt=${d}..${p}
    for nshot in 1 2 5 10
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot
    done
  done
done

#===========================
# Method: 'Src_Tgt'


method="Src_Tgt"

for d in nexus4 s3 s3mini lgwatch
do
  for p in a b c d e f
  do
    tgt=${d}..${p}
    for nshot in 1 2 5 10
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot
    done
  done
done

##===========================
## Method: 'TrC'


method="TrC"

for d in nexus4 s3 s3mini lgwatch
do
  for p in a b c d e f
  do
    tgt=${d}..${p}
    for nshot in 1 2 5 10
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --load_checkpoint_path log/hhar_scaled/Src/src_rest/tgt_${tgt}/run_test/cp/
    done
  done
done


##===========================
## Method: 'PN'


method="PN"

for d in nexus4 s3 s3mini lgwatch
do
  for p in a b c d e f
  do
    tgt=${d}..${p}
    for nshot in 5
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
    done
  done
done

##===========================
## Method: 'MAML'


method="MAML"

for d in nexus4 s3 s3mini lgwatch
do
  for p in a b c d e f
  do
    tgt=${d}..${p}

    for nshot in 5
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
    done
  done
done

##===========================
## Method: 'MetaSense'

method="MetaSense"

for d in nexus4 s3 s3mini lgwatch
do
  for p in a b c d e f
  do
    tgt=${d}..${p}
    for nshot in 5
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
    done
  done
done


##################################################
################## Dataset: DSA ##################
##################################################

dataset="dsa_scaled"
model="DSA_model"
meta_lr="0.15"



#===========================
# Method: 'Src'
method="Src"
for p in 1 2 3 4 5 6 7 8
do
  for po in T RA LA RL LL
  do
    tgt=p${p}.${po}
    python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test  --src rest --train 1 --model $model
  done
done

#===========================
# Method: 'Tgt'

method="Tgt"
for p in 1 2 3 4 5 6 7 8
do
  for po in T RA LA RL LL
  do
    tgt=p${p}.${po}
    for nshot in 1 2 5 10
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot
    done
  done
done

#===========================
# Method: 'Src_Tgt'


method="Src_Tgt"

for p in 1 2 3 4 5 6 7 8
do
  for po in T RA LA RL LL
  do
    tgt=p${p}.${po}
    for nshot in 1 2 5 10
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot
    done
  done
done

##===========================
## Method: 'TrC'


method="TrC"

for p in 1 2 3 4 5 6 7 8
do
  for po in T RA LA RL LL
  do
    tgt=p${p}.${po}
    for nshot in 1 2 5 10
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --load_checkpoint_path log/dsa_scaled/Src/src_rest/tgt_${tgt}/run_test/cp/
    done
  done
done


##===========================
## Method: 'PN'


method="PN"

for p in 1 2 3 4 5 6 7 8
do
  for po in T RA LA RL LL
  do
    tgt=p${p}.${po}
    for nshot in 5
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
    done
  done
done

##===========================
## Method: 'MAML'


method="MAML"

for p in 1 2 3 4 5 6 7 8
do
  for po in T RA LA RL LL
  do
    tgt=p${p}.${po}

    for nshot in 5
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
    done
  done
done

##===========================
## Method: 'MetaSense'

method="MetaSense"

for p in 1 2 3 4 5 6 7 8
do
  for po in T RA LA RL LL
  do
    tgt=p${p}.${po}
    for nshot in 5
    do
      python main.py --dataset $dataset --method $method --tgt $tgt --epoch 200 --log_suffix run_test_${nshot}shot  --src rest --train 1 --model $model --nshot $nshot --lr $meta_lr
    done
  done
done
