from utils import *

list_train_data_mat = split_img('./datasets/chikusei/chikusei/chikusei.mat', 400, 200, is_rgb=False, is_test=False,
                                img_name='chikusei')
list_train_data_rgb = split_img('./datasets/chikusei/Chikusei.jpg', 400, 200, is_rgb = True,is_test = False)
print(len(list_train_data_rgb), len(list_train_data_mat))
assert len(list_train_data_mat) == len(list_train_data_rgb)

list_validation_mat_data = []
list_validation_rgb_data = []
img_length = len(list_train_data_mat)
for i in range(int(img_length * 0.2)):
    rand_number = random.randint(0, img_length - i - 1)
    img_mat = list_train_data_mat[rand_number]
    img_rgb = list_train_data_rgb[rand_number]

    list_validation_mat_data.append(img_mat)
    list_validation_rgb_data.append(img_rgb)

    del list_train_data_mat[rand_number]
    del list_train_data_rgb[rand_number]

assert len(list_validation_mat_data) == len(list_validation_rgb_data)

for i in range(len(list_train_data_mat)):
    np.save('./datasets/chikusei/train/mat/chikusei_train_patch_{}.npy'.format(i), list_train_data_mat[i])
    np.save('./datasets/chikusei/train/rgb/chikusei_train_patch_{}.npy'.format(i), list_train_data_rgb[i])

for i in range(len(list_validation_mat_data)):
    np.save('./datasets/chikusei/validation/mat/chikusei_validation_patch_{}.npy'.format(i), list_validation_mat_data[i])
    np.save('./datasets/chikusei/validation/rgb/chikusei_validation_patch_{}.npy'.format(i), list_validation_rgb_data[i])

# np.save('./datasets/chikusei/test/chikusei_test_patch.npy', test_img)
