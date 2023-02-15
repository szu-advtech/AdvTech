./build/run_video_slam -v ./orb_vocab/orb_vocab.dbow2 -t ./tsdf_config.yaml -m aist_living_lab_1/video.mp4 -c aist_living_lab_1/config.yaml --frame-skip 3 --no-sleep --map-db map.msg
cd /home/code/PotreeConverter/build/PotreeConverter
./PotreeConverter /home/code/openvslam-comments/tsdf_new.ply -o /home/code/potree -p tsdf --overwrite
