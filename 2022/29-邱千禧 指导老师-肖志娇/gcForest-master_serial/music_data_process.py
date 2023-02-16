import librosa
import numpy as np
import os

#特征提取参考： https://blog.csdn.net/weixin_42129113/article/details/112245106
# 读文件、代码参考：https://blog.csdn.net/MC_XY/article/details/121249949
#metal pop reggae rock blues classical country disco


def get_feature(y):

    rmse = librosa.feature.rms(y=y)
    label_features_line = rmse

    # 色度频率   12 ,1292
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    shape = np.shape(chroma_stft)
    for i in range(shape[0]):
        label_features_line = np.concatenate([label_features_line, chroma_stft[i, :].reshape(1, -1)], axis=1)

    # 光谱质心
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    label_features_line = np.concatenate([label_features_line, spec_cent], axis=1)

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    label_features_line = np.concatenate([label_features_line, spec_bw], axis=1)

    # 光谱衰减
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    label_features_line = np.concatenate([label_features_line, rolloff], axis=1)

    # 过零率
    zcr = librosa.feature.zero_crossing_rate(y)
    label_features_line = np.concatenate([label_features_line, zcr], axis=1)

    # 梅尔频率倒谱系数  20.1292
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    for j in range(np.shape(mfcc)[0]):
        label_features_line = np.concatenate([label_features_line, mfcc[j, :].reshape(1, -1)], axis=1)

    data.append(label_features_line[0, :].tolist())
    target.append(g)



genres = 'metal pop reggae rock blues classical country disco'.split()

label2id = {genre:i for i,genre in enumerate(genres)}
id2label = {i:genre for i,genre in enumerate(genres)}
print(label2id)


data=[]
target=[]


for g in genres:
    print(g)
    for filename in os.listdir(f'./genres/{g}/'):
        songname = f'./genres/{g}/{filename}'


        #load函数就是用来读取音频的。当然，读取之后，转化为了numpy的格式储存，而不再是音频的格式了。
        #y: 音频的信号值，类型是ndarray
        #sr: 采样率,默认22050，但是最终维度就会有4w多。
        y, sr = librosa.load(songname, sr=220,mono=True, duration=30)
        #噪声增强，0.004是噪声因子
        y_noise=y+0.004*np.random.normal(loc=0,scale=1,size=len(y))
        y_noise2 = y + 0.002 * np.random.normal(loc=0, scale=1, size=len(y))
        #波形移位
        y_roll=np.roll(y,int(sr//2))
        y_roll2 = np.roll(y, int(sr // 4))
        #音高修正
        y_high=librosa.effects.pitch_shift(y,sr,n_steps=3,bins_per_octave=24)
        y_high2 = librosa.effects.pitch_shift(y, sr, n_steps=-3, bins_per_octave=24)
        get_feature(y)
        get_feature(y_high)
        get_feature(y_roll)
        get_feature(y_noise)
        get_feature(y_high2)
        get_feature(y_roll2)
        get_feature(y_noise2)






data=np.array(data,dtype="float64")
target=np.array(target)


np.savez('220_datasets_7_plus_old.npz',data=data,target=target)
