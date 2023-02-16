import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge

np.random.seed(0)

def get_crop():
	df_crop =pd.read_csv('data/FAOSTAT_data_main_crop_yield.csv')
	df_reg =pd.read_csv('data/regional_code.csv')

	# print(df_crop.head())

	country_code = df_reg['Country Code']
	iso_code = df_reg['ISO2 Code']

	code_mapping = {}
	for c_code, i_code in zip(country_code, iso_code):
		code_mapping[c_code] = i_code

	value = df_crop['Value']
	country_code = df_crop['Area Code']
	year = df_crop['Year']
	print(len(value))

	items = df_crop['Item']
	

	countries_data = pd.read_csv('data/countries_cords.csv')

	names = np.array(countries_data['Alpha-2 code'])
	lats = countries_data['Latitude']
	lons = countries_data['Longitude']
	lats = np.array([float(val.strip()[1:-1]) for val in lats])
	lons = np.array([float(val.strip()[1:-1]) for val in lons])

	names_dict = {}
	for idx, name in enumerate(names):
		names_dict[name.strip()[1:-1]] = (lons[idx], lats[idx])

	indices = []
	lats_mapped = []
	lons_mapped = []
	for idx, c_code in enumerate(country_code):
		if c_code in code_mapping and code_mapping[c_code] == code_mapping[c_code] and code_mapping[c_code] in names_dict:
			indices.append(idx)
			iso_code = code_mapping[c_code]
			lons_mapped.append(names_dict[iso_code][0])
			lats_mapped.append(names_dict[iso_code][1])

	indices = np.array(indices)
	value = np.array(value[indices]).astype(np.float32)
	year = np.array(year[indices]).astype(np.float32)
	items = np.array(items[indices])

	print(value[:10])
	print(year[:10])

	print(len(value), len(year), len(lons_mapped), len(lats_mapped), len(items))
	return items, year, value, lons_mapped, lats_mapped

def read_csv(fname):
	df = pd.read_csv(fname)
	timestamp = np.array(df['timestamp']).astype(np.float32)
	latitude = np.array(df['latitude']).astype(np.float32)
	longitude = np.array(df['longitude']).astype(np.float32)

	temperature = np.array(df['temperature']).astype(np.float32)
	wind_speed = np.array(df['wind_speed']).astype(np.float32)
	precipitation = np.array(df['precipitation']).astype(np.float32)
	# raining = np.array(df['raining']).astype(np.float32)

	stn = np.array(df['stn']).astype(np.float32)
	temp_dict = {}
	wind_dict = {}
	prcp_dict = {}
	lon_dict = {}
	lat_dict = {}
	for i in range(len(stn)):
		temp_dict[(stn[i], timestamp[i])] = temperature[i]
		wind_dict[(stn[i], timestamp[i])] = wind_speed[i]
		prcp_dict[(stn[i], timestamp[i])] = precipitation[i]
		lon_dict[stn[i]] = longitude[i]
		lat_dict[stn[i]] = latitude[i]

	return timestamp, stn, lat_dict, lon_dict, temp_dict, wind_dict, prcp_dict

def read_raining_csv(fname):
	df = pd.read_csv(fname)
	raining = np.array(df['c']).astype(np.float32)
	stn = np.array(df['stn']).astype(np.float32)
	year = np.array(df['year']).astype(np.float32)

	raining_year_dict = {}
	for i in range(len(stn)):
		raining_year_dict[(stn[i], year[i])] = raining[i]

	return raining_year_dict 


def combine_dict(dict1, dict2):
	for key in dict2:
		dict1[key] = dict2[key]
	for key in dict1:
		if dict1[key] != dict1[key]:
			dict1[key] = 0.0
	return dict1

items, year_crop, value, lons_mapped, lats_mapped = get_crop()
locations_mapped = np.array([[lon, lat] for lon, lat in zip(lons_mapped, lats_mapped)])

timestamp, station, lat_dict, lon_dict, temp_dict, wind_dict, prcp_dict = read_csv("data/19.csv")
print(len(np.unique(station)))


timestamp_20, station_20, lat_dict_20, lon_dict_20, temp_dict_20, wind_dict_20, prcp_dict_20 = read_csv("data/20.csv")
print(len(np.unique(station_20)))

final_stations = np.load("data/final_stations.npy")

raining_year_dict = read_raining_csv('data/19_raining.csv')
raining_year_dict_20 = read_raining_csv('data/20_raining.csv')

c = 0

years = np.arange(1981, 2000)
years_20 = np.arange(2000, 2020)

new_stations = []
for stn in final_stations:
	flag = 1
	for year in years:
		if (stn, float(year)) not in raining_year_dict:
			flag = 0
			break
	if flag == 1 :
		for year in years_20:
			if (stn, float(year)) not in raining_year_dict_20:
				flag = 0
				break
	c += flag
	if flag == 1:
		new_stations.append(stn)

print("the total count is ", c)
final_stations = np.array(new_stations)
print("the len of new stations is ", len(final_stations))

temp_dict = combine_dict(temp_dict, temp_dict_20)
wind_dict = combine_dict(wind_dict, wind_dict_20)
prcp_dict = combine_dict(prcp_dict, prcp_dict_20)
raining_year_dict = combine_dict(raining_year_dict, raining_year_dict_20)

lons_final = np.array([lon_dict[stn] for stn in final_stations])
lats_final = np.array([lat_dict[stn] for stn in final_stations])

locations = []
for stn in final_stations:
	locations.append([lon_dict[stn], lat_dict[stn]])

locations = np.array(locations)

distances = euclidean_distances(locations_mapped, locations)
print(distances.shape)
indices = np.argmin(distances, axis=-1)
stations_mapped = final_stations[indices]

print(stations_mapped.shape)

print(np.unique(year_crop))

all_temp = [temp_dict[(stn, year)] for (stn, year) in zip(stations_mapped, year_crop) if year > 1980]
all_wind = [wind_dict[(stn, year)] for (stn, year) in zip(stations_mapped, year_crop) if year > 1980]
all_prcp = [prcp_dict[(stn, year)] for (stn, year) in zip(stations_mapped, year_crop) if year > 1980]
all_rain = [raining_year_dict[(stn, year)] for (stn, year) in zip(stations_mapped, year_crop) if year > 1980]

min_temp, max_temp = np.min(all_temp), np.max(all_temp)
min_wind, max_wind = np.min(all_wind), np.max(all_wind)
min_prcp, max_prcp = np.min(all_prcp), np.max(all_prcp)
min_rain, max_rain = np.min(all_rain), np.max(all_rain)

# print(min_temp, max_temp, min_wind, max_wind, min_prcp, max_prcp, min_rain, max_rain)
def make_dataset(items, year_crop, value, stations_mapped, temp_dict, wind_dict, prcp_dict, raining_year_dict):
	crop_data_dict = defaultdict(list)
	crop_label_dict = defaultdict(list)
	crops = np.unique(items)
	temp_min = 200.0
	temp_max = -100.0
	prcp_min = 200.0
	prcp_max = -100.0
	wind_min = 200.0
	wind_max = -100.0

	for crop in crops:
		for idx, item in enumerate(items):
			if item == crop:
				year = year_crop[idx]
				if year > 1980 and value[idx] == value[idx]:
					stn = stations_mapped[idx]
					temp = (temp_dict[(stn, year)] - min_temp)/(max_temp - min_temp)
					wind = (wind_dict[(stn, year)] - min_wind)/(max_wind - min_wind)
					prcp = (prcp_dict[(stn, year)] - min_prcp)/(max_prcp - min_prcp)
					rain = (raining_year_dict[(stn, year)] - min_rain)/(max_rain - min_rain)
					crop_data_dict[item].append([temp, wind, prcp, rain])
					crop_label_dict[item].append(value[idx])

	for crop in crops:
		crop_data_dict[crop] = np.array(crop_data_dict[crop])
		crop_label_dict[crop] = np.array(crop_label_dict[crop])
		print(crop, crop_data_dict[crop].shape)

	return crop_data_dict, crop_label_dict, crops

crop_data_dict, crop_label_dict, crops = make_dataset(items, year_crop, value, stations_mapped, temp_dict, 
							wind_dict, prcp_dict, raining_year_dict)

def get_test_data(idx):
	data = []
	temp = (all_temp - min_temp)/(max_temp - min_temp)
	wind = (all_wind - min_wind)/(max_wind - min_wind)
	prcp = (all_prcp - min_prcp)/(max_prcp - min_prcp)
	rain = (all_rain - min_rain)/(max_rain - min_rain)
	all_datas = [temp, wind, prcp, rain]
	x_val = []
	for i in range(50):
		temp = []
		for j in range(4):
			if j == idx:
				temp.append(all_datas[j][i])
				x_val.append(temp[-1])
			else:
				temp.append(all_datas[j][0])
		data.append(temp)
	data = np.array(data)
	x_val = np.array(x_val)
	indices = np.argsort(x_val)
	x_val = x_val[indices]
	data = data[indices]

	return data, x_val

models = {}
crops = ['Rice, paddy', 'Wheat']

for crop in crops:
	models[crop] = KernelRidge(alpha=1.0)
	x = crop_data_dict[crop][:3000]
	y = crop_label_dict[crop][:3000]
	y = (y - np.min(y))/(np.max(y) - np.min(y))
	print(crop, x.shape, y.shape, np.min(x), np.max(x), np.min(y), np.max(y))
	models[crop].fit(x, y)


def get_new_data(stn, years):
	data = []
	for year in years:
		temp = (temp_dict[(stn, year)] - min_temp)/(max_temp - min_temp)
		wind = (wind_dict[(stn, year)] - min_wind)/(max_wind - min_wind)
		prcp = (prcp_dict[(stn, year)] - min_prcp)/(max_prcp - min_prcp)
		rain = (raining_year_dict[(stn, year)] - min_rain)/(max_rain - min_rain)
		data.append([temp, wind, prcp, rain])
	data = np.array(data)
	return data

def compute_divergence(stn):
	year1 = np.arange(1981, 2000)
	year2 = np.arange(2000, 2020)

	data1 = get_new_data(stn, year1)
	data2 = get_new_data(stn, year2)

	year_all = np.array(list(year1) + list(year2))
	np.random.shuffle(year_all)
	year_m = year_all[:len(year1)]
	datam = get_new_data(stn, year_m)
	# print(len(year1), len(year2), len(year_m))

	loss = []
	for crop in crops:
		util1 = np.sum(models[crop].predict(data1))
		util2 = np.sum(models[crop].predict(data2))
		utilm = np.sum(models[crop].predict(datam))
		l1 = -util1
		l2 = -util2
		lm = -utilm

		loss.append(lm - min(l1, l2))
      
	return np.min(loss)


divergences = []

for stn in final_stations:
	divergences.append(compute_divergence(stn))

divergences = np.array(divergences)
longitudes = np.array(lons_final)
latitudes = np.array(lats_final)


print(divergences.shape)
np.save("data/divergences_crop_final_ridge.npy", divergences)
np.save("data/latitudes_crop_final_ridge.npy", latitudes)
np.save("data/longitudes_crop_final_ridge.npy", longitudes)







