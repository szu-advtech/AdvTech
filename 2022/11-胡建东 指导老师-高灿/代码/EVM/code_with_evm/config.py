tailsize = 75 #尾部尺寸大小，实验设置是75，这里用的是80    75
cover_threshold = 0.95 #覆盖的阈值，实验设置是0.5，这里用的是0.8   0.8
num_to_fuse = 4   #k=4，实验时也是4
margin_scale=0.5  #边界的一个缩放，实验中最小二半是0.5，这里用的是0.8   0.5
ot = 0.05  #最后的概率阈值设置（置信度），如果小于阈值，则会是未知类即为99  0.1,实验是0.05