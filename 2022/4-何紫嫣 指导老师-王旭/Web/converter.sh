sleep 3
# ps -ef | grep run_video_slam | grep -v grep | awk '{print $2}' | xargs kill -9
# cd /home/code/PotreeConverter/build/PotreeConverter
/home/code/PotreeConverter/build/PotreeConverter/PotreeConverter /home/code/web/upload/tsdf_new.ply -o /home/code/web -p testnew --overwrite
cp /home/code/web/getresult.js /home/code/web/upload/getresult.js