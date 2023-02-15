import os
import skeleton

root = "./dataset/deepfashionHD/"
root1 = "./data/deepfashionHD/"
if __name__ == '__main__':
    with open('./data/train.txt','r') as fd:
        image_path_list = fd.readlines()

    fo = open("./data/mytrain.txt","w")

    for image_path in image_path_list:
        image_path = image_path.strip()
        skeleton_path1 = image_path.replace(".jpg","_candidate.txt")
        skeleton_path2 = image_path.replace(".jpg","_subset.txt")
        skeleton_path1 = skeleton_path1.replace("img","pose")
        skeleton_path2 = skeleton_path2.replace("img","pose")
        fo.write(image_path+","+skeleton_path1.format("candidate")+","+skeleton_path2.format("subset")+"\n")
    fo.close()

    with open('./data/mytrain.txt','r') as fd2:
        image_path_list = fd2.readlines()

    for image_paths in image_path_list:
        image_paths = image_paths.strip()
        path_candidate = image_paths.split(",")[1]
        path_subset = image_paths.split(",")[2]
        poseimg = skeleton.get_label_tensor(root1 + path_candidate,root1 + path_subset)
        save_path = root + path_subset
        save_path = save_path.replace("\\","/")
        save_path = save_path.replace("pose","pose_img")
        save_path = save_path.replace("_subset.txt",".jpg")
        print(save_path)
        head,tail = os.path.split(save_path)
        print(head,tail)
        if not os.path.exists(head):
            os.makedirs(head)
        poseimg.save(save_path)