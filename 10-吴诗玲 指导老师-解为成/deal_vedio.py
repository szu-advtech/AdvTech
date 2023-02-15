import numpy as np
import pandas as pd
import os


def reduce_frames_AVEC2014(openface_dir, save_dir, detail_dir):
    listdir = os.listdir(openface_dir)
    task_name = ['Freeform', 'Northwind']
    for i in listdir:
        for j in task_name:
            print(i, j)
            openface_files = os.path.join(openface_dir, i, j)
            files = os.listdir(openface_files)
            save_path = os.path.join(save_dir, i, j)
            os.makedirs(save_path, exist_ok=True)
            detail_path = os.path.join(detail_dir, i)
            os.makedirs(detail_path, exist_ok=True)
            deatil_file = detail_path + "/" + j + ".csv"
            detail = pd.DataFrame(columns=['file', 'before_frames', 'now_frames', 'now_fre'])
            index = 0

            for file in files:
                if os.path.splitext(file)[1] == '.csv':
                    openface_file = os.path.join(openface_dir, i, j, file)
                    content = pd.read_csv(openface_file)
                    frames = content.shape[0]
                    time = frames / 30
                    content.drop(content[content[' success'] == 0].index, inplace=True)
                    os.makedirs(save_path, exist_ok=True)
                    now_frames = content.shape[0]
                    fre = now_frames / time
                    detail.loc[index] = [file, frames, now_frames, fre]
                    index += 1
                    save_file = os.path.join(save_path, file)
                    content.to_csv(save_file, index=False)
            detail.to_csv(deatil_file, index=False)


def reduce_frames_AVEC2019(openface_dir, save_dir, detail_dir,split_dir):
    detail = pd.DataFrame(columns=['file', 'before_frames', 'now_frames', 'now_fre'])
    splits=os.listdir(split_dir)
    os.makedirs(detail_dir, exist_ok=True)
    for i in splits:
        index = 0
        if i[-9:]=="split.csv":
            dir=i[:-10]
            detail_file=os.path.join(detail_dir,dir+".csv")
            part=os.path.join(split_dir,i)
            info=pd.read_csv(part,header=0)
            person=info['Participant_ID']
            save_path=os.path.join(save_dir,dir)
            os.makedirs(save_path, exist_ok=True)
            for p in person:
                p=str(p)
                pp=p+"_P"
                j = p + "_OpenFace2.1.0_Pose_gaze_AUs.csv"
                openface_file = os.path.join(openface_dir, pp, "features", j)
                save_file = os.path.join(save_path,j)
                content = pd.read_csv(openface_file)
                frames = content.shape[0]
                time = frames / 30
                content.drop(content[content['success'] == 0].index, inplace=True)
                now_frames = content.shape[0]
                fre = now_frames / time
                detail.loc[index] = [i, frames, now_frames, fre]
                index += 1
                content.to_csv(save_file, index=False)
            detail.to_csv(detail_file, index=False)

if __name__ == "__main__":
    reduce_frames_AVEC2019(r"F:\AVEC2019\extracted",r"F:\AVEC2019\Openface_delete","F:\AVEC2019\detail",r'F:\AVEC2019\labels')
    #reduce_frames_AVEC2014(r"AVEC2014/Openface", r"AVEC2014/Openface_delete", r'AVEC2014/detail')
