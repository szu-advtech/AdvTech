import matplotlib.pyplot as plt


waveNum = [i for i in range(1, 151)]
output_intensity = []
target_intensity = []
output_path = './coodinate/building_output.txt'
target_path = './coodinate/building_target.txt'
with open(output_path) as f:
    for line in f:
        output_intensity.append(float(line.split()[1]))

with open(target_path) as f:
    for line in f:
        target_intensity.append(float(line.split()[1]))

plt.plot(waveNum, output_intensity, label='output')
plt.plot(waveNum, target_intensity, label='target')
plt.legend()
plt.xlabel('wavelength number')
plt.ylabel('normalized intensity')
plt.title('building')
plt.show()



# print('waveNum:', waveNum)
# print('intensity', intensity)
# print('intensity.len:', len(intensity))











