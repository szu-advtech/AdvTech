cd build/
cmake \
    -DBUILD_WITH_MARCH_NATIVE=ON \
    -DUSE_PANGOLIN_VIEWER=OFF \
    -DUSE_SOCKET_PUBLISHER=OFF \
    -DUSE_STACK_TRACE_LOGGER=ON \
    -DBOW_FRAMEWORK=DBoW2 \
    -DBUILD_TESTS=ON \
    ..
make -j8
cd ../
./build/run_video_slam -v ./orb_vocab/orb_vocab.dbow2 -t ./tsdf_config.yaml -m aist_living_lab_3/video.mp4 -c aist_living_lab_3/config.yaml --frame-skip 3 --no-sleep --map-db map.msg
