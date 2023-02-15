# cmake -DCMAKE_BUILD_TYPE=Release ..

dataset=(linear seg1 seg10 normal books_200M_uint64 fb_200M_uint64 osmc_200M_uint64 wiki_ts_200M_uint64)
for ds in ${dataset[@]}
do
    echo ">>>>>>>>>> $ds: 时延 <<<<<<<<<<"
    ./build/main_latency ./dataset/$ds.csv 0 1
done

for ds in ${dataset[@]}
do
    echo ">>>>>>>>>> $ds: 不同静态数据集吞吐量 <<<<<<<<<<"
    ./build/main_goodput ./dataset/$ds.csv 0 1
done
