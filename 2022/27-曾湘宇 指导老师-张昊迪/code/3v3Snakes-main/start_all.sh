nohup python -u actor.py --num_replicas=20 > ./log/actor.log &
nohup python -u learner.py --pool_size=16384 --batch_size=16384 > ./log/learner.log &
