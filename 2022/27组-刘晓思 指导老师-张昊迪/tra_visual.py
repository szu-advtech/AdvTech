import folium

tra_list = []

min_lat = 30.65541009760488
max_lat = 30.68238970863773
min_lon = 104.03969948534711
max_lon = 104.07480006725845

with open('data/cd_test_trajs.txt', 'r') as f:

    for line in f.readlines():
        attrs = line.rstrip().split(' ')
        i = 0
        p_list = []
        while i < len(attrs)-1:
            lat = float(attrs[i])
            i += 1
            lng = float(attrs[i])
            i += 1

            if lat < min_lat or lng < min_lon or lat > max_lat or lng > max_lon:
                continue

            p_list.append([lat, lng])

        if len(p_list) > 0:
            tra_list.append(p_list)


map = folium.Map(location=[(max_lat+min_lat)/2, (max_lon+min_lon)/2],
               zoom_start=14.8,
               tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
               attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
               )

for t in tra_list:
    if len(t) > 0:
        folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
            t,  # 将坐标点连接起来
            weight=3,  # 线的大小为3
            color='#FF9900',  # 线的颜色为橙色
            opacity=0.8  # 线的透明度
        ).add_to(map)  # 将这条线添加到刚才的区域m内

folium.Rectangle(
        bounds=((min_lat, min_lon), (max_lat, max_lon)),
        color='gray',
        fill=False
    ).add_to(map)

map.save('result/raw_traj.html')

