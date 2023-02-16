import os

pathname = "./audio/"
# dirpath:文件夹路径
# dirnames:子文件夹名字
# filenames:子文件夹内文件名
for dirpath, dirnames, filenames in os.walk(pathname):
    for filename in filenames:
        # print(filename)
        audio_path = dirpath + filename
        video_fname = filename.split('.')[0] + '_MeshTalk.mp4'
        os.system('python animate_face.py --audio_file "{0}" --output "./output/{1}"'.format(audio_path, video_fname))
