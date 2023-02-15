import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from itertools import chain
from scipy import interpolate
from mpl_toolkits.basemap import maskoceans
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

np.random.seed(0)
cmap = sns.cubehelix_palette(as_cmap=True)

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    # lats = m.drawparallels(np.linspace(-90, 90, 13))
    # lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # # keys contain the plt.Line2D instances
    # lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    # lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    # all_lines = chain(lat_lines, lon_lines)
    
    # # cycle through these lines and set the desired style
    # for line in all_lines:
    #     line.set(linestyle='-', alpha=0.3, color='w')

# fig = plt.figure(figsize=(10, 10), edgecolor='w')


m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
draw_map(m)

divergences = np.load("data/divergences_crop_final_ridge.npy")
latitudes = np.load("data/latitudes_crop_final_ridge.npy")
longitudes = np.load("data/longitudes_crop_final_ridge.npy")

indices = np.arange(len(divergences))
np.random.shuffle(indices)
points = 3000

latitudes = latitudes[indices][:points]
longitudes = longitudes[indices][:points]
divergences = divergences[:points]

longitudes,latitudes = m(longitudes, latitudes)


points = np.array([[val1, val2] for (val1, val2) in zip(longitudes, latitudes)])
print("interpolating")
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points)



print(np.min(divergences), np.max(divergences))

lons_all = np.arange(-180, 181, 2)
lats_all = np.arange(-60, 91, 2)

lons_all,lats_all = m(lons_all, lats_all)

lons_all, lats_all = np.meshgrid(lons_all, lats_all)

z_new = np.zeros(lons_all.shape)

new_points = []

for i in range(len(lons_all)):
    for j in range(len(lons_all[0])):
        new_points.append([lons_all[i][j], lats_all[i][j]])

new_points = np.reshape(np.array(new_points), [-1, 2])
print("new points shape ", new_points.shape)

_, indices = nbrs.kneighbors(new_points)
z_new = []
indices = np.array(indices)
print("indices shape ", indices.shape)

z_new = divergences[indices]
z_new = np.mean(z_new, axis=-1)
z_new = np.reshape(z_new, lons_all.shape)


nc_new = maskoceans(lons_all,lats_all,z_new)
m.contourf(lons_all,lats_all,nc_new, cmap = cmap)


plt.colorbar(label='divergence', shrink = 0.5)
plt.savefig("plotting_final_crop.png")

