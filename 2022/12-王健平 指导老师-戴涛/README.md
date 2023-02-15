```python
预处理阶段：
生成测试集：
python prepare.py # 训 练 集 图 片 所 在 文 件 夹 \
--images-dir "dataset_train/eSR" \
# 训 练 集h5文 件 输 出 文 件 夹 \
--output-path "h5_file/91-image/train_x3.h5" \
--scale 3 # 放 大 倍 数
                
生成测试集：
python prepare.py 
# 测 试 集 图 片 所 在 文 件 夹 \
--images-dir "dataset_eval/eSR" \
# 测 试 集h5文 件 输 出 文 件 夹 \
--output-path "h5_file/91-image/eval_x3.h5" \
--scale 3 # 放 大 倍 数 \
--eval # 测 试 集 标 志
训练阶段：
python train.py 
--net "edgeSR_TR_ECBSR" # 选 用 的 网 络 模 型 \
# h5格 式 训 练 集 所 在 位 置 \
--train-file "h5_file/91-image/train_x3.h5" \
# h5格 式 测 试 集 所 在 位 置 \
--eval-file "h5_file/91-image/eval_x3.h5" \
# 输 出 保 存 模 型 参 数 的 文 件 夹 \
--outputs-dir "outputs" \
# 选 填 \
--gpu-id 0 # 使 用 的gpu \
--scale 3 # 放 大 的 倍 数 \
--lr 1e-3 # 学 习 率 \
--batch-size 16 # 每 次 处 理 的 批 次 大 小 \
--num-epochs 20 # 训 练 轮 次 \
--num-workers 16 # 预 加 载 的 线 程 个 数 \
--seed 123 # 随 机 数 种 子 \
--group-conv 1 # 是 否 使 用 分 组 卷 积
测试和评估阶段：
python test.py --net "edgeSR_MAX" # 模 型 网 络 \
--outputs-dir "outputs/edgeSR_MAX_x3" # 输 出 文 件 夹 \
--weights-file "outputs/edgeSR_MAX_x3/best.pth" # 模 型 参 数 \
--gpu-id 0 # gpu \
--image-dir-test "dataset/Set5" # 推 理 图 片 文 件 夹 \
--image-dir-eval "dataset/Set5" # 评 估 图 片 文 件 夹 \
--scale 3 # 放 大 的 倍 
```

