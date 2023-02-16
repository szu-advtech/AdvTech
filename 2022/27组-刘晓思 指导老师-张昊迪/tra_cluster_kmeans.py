from sklearn.cluster import KMeans
import numpy as np
import folium

x = []
y = []
max_p = 0
num = 0

min_lat = 30.65541009760488
max_lat = 30.68238970863773
min_lon = 104.03969948534711
max_lon = 104.07480006725845

with open('data/cd_test_trajs.txt', 'r') as f:

    for line in f.readlines():
        attrs = line.rstrip().split(' ')
        i = 0
        lats = []
        lngs = []
        while i < len(attrs):
            lat = float(attrs[i])
            i += 1
            lng = float(attrs[i])
            i += 1

            lats.append(lat)
            lngs.append(lng)

        if len(lats) <= 0:
            continue
        if len(lats) > max_p:
            max_p = len(lats)
        num += 1
        x.append(lats)
        y.append(lngs)

print('max_points:{}'.format(max_p))

traj = np.zeros((num, max_p,2))
for i in range(len(lats)):
    traj[i, 0:len(x[i]), 0] = x[i]
    traj[i, 0:len(y[i]), 1] = y[i]
# traj[:,:,0] = x
# traj[:,:,1] = y
traj = traj.reshape((num, max_p*2))

print('shape trajs:{}'.format(traj.shape))

# init：有三个可选值：‘k-means++’、‘random’、或者传递一个ndarray向量。
# １）‘k-means++’ 用一种特殊的方法选定初始质心从而能加速迭代过程的收敛
# ２）‘random’ 随机从训练数据中选取初始质心。
# ３）如果传递的是一个ndarray，则应该形如 (n_clusters, n_features) 并给出初始质心。
# 默认值为‘k-means++’。

km = KMeans(n_clusters=6, init='k-means++')#构造聚类器
km.fit(traj)#聚类
labels = km.labels_ #获取聚类标签
# 获取质心（聚类中心）
centers = km.cluster_centers_
print('label :{}, len:{}'.format(labels, len(labels)))

centers = centers.reshape((6, max_p, 2))
# print('centers:{}'.format(centers[2][:20]))

cen_traj = []

for t in centers:
    # print('t:{}'.format(t))
    for i in range(len(t)):
        if t[i][0] < 30:
            cen_traj.append(t[:i])
            break


# print(cen_traj[5])



map = folium.Map(location=[(max_lat+min_lat)/2, (max_lon+min_lon)/2],
               zoom_start=14.8,
               tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
               attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
               )

for t in cen_traj:
    if len(t) > 0:
        folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
            t,  # 将坐标点连接起来
            weight=3,  # 线的大小为3
            color='blue',  # 线的颜色为橙色
            opacity=0.8  # 线的透明度
        ).add_to(map)  # 将这条线添加到刚才的区域m内

folium.Rectangle(
        bounds=((min_lat, min_lon), (max_lat, max_lon)),
        color='gray',
        fill=False
    ).add_to(map)

map.save('result/kmeans_traj.html')

