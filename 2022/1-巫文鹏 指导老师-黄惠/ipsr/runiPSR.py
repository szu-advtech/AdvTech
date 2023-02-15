import os

data_path = "./data"
mode = ['default', 'without_area', 'distance_weighted', 'ball_query']

plyfile = [os.path.join(f) for f in os.listdir(data_path) if f.endswith('ply')]

output_dir = './Reconed_model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for i in range(len(plyfile)):
    for j in range(1, 5):
        # for iter in range(1, 10+1):
        args = './ipsr'

        args += ' --in ' + data_path + '/' + plyfile[i]
        args += ' --mode '+ str(j)
        args += ' --out '+ output_dir + '/' + mode[j-1] + '_' +  plyfile[i]

        print(args)
        out = os.popen(args)
        print(out.read())

print("script end")










