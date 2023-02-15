cd build

for PY_VERSION in 3.5 3.6 3.7 3.8 3.9
do
	CXX=/usr/bin/clang++-11 CC=/usr/bin/clang-11 cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DRAISIM_MATLAB=OFF -DRAISIM_PY=ON -DRAISIM_EXAMPLE=OFF -DRAISIM_DOC=OFF -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python${PY_VERSION}
    make -j
done

CXX=/usr/bin/clang++-11 CC=/usr/bin/clang-11 cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DRAISIM_EXAMPLE=ON -DRAISIM_DOC=ON -DRAISIM_PY=OFF -DRAISIM_MATLAB=ON -DPYTHON_EXECUTABLE:FILEPATH=/home/jemin/anaconda3/envs/raisim/bin/python
make -j
