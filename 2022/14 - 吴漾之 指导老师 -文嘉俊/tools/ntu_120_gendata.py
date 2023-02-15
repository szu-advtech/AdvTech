import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30

def print_toolbar(rate, annotation=''):  # Output: annotation[- -]
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")  # After loading a module, change the line


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    if ignored_sample_path != None:  # have this path
        with open(ignored_sample_path, 'r') as f:  # Open in read-only form
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()  # ['line1.skeleton','line2.skeleton'...]
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    dir_list = os.listdir(data_path)  # all files and folders in the specify path
    dir_list.sort()  # Sort for file names
    for filename in dir_list:
        if filename in ignored_samples:
            continue  # Skip directly in the ignored_sample, otherwise continue the current loop
        """
        Get the action category ,person id and camera number from the file name and convert  to 'int'
        For example,S001C001P001R001A004,cut the three fields after the found letter and convert them to 'int'
        """
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining  # Training sample
        elif part == 'val':
            issample = not (istraining) # Flip the bool variable
        else:
            raise ValueError()

        if issample:  # is training sample
            sample_name.append(filename)  # The training samples are stored in a list
            sample_label.append(action_class - 1)  # Label people from 0

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:  # To write pictures or video or audio in binary form
        pickle.dump((sample_name, list(sample_label)), f)  # Use 'pickle' module to save data objects to file
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(

        # Using the function 'np.memmap' and passing in a file path, data type, shape, and file pattern,
        # that can create a new memmap object, treats very large binary data files on disk as an array in memory

        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',  # Read and write, created if the file does not exist
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))  # A joint generally has three coordinate features

    for i, s in enumerate(sample_name):  # Iterate over a collection object and get the index position of the current element
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))  # get 'rate' and 'annotation'
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)  # Pass in sample_name, maximum number of people, number of joints
        fp[i, :, 0:data.shape[1], :, :] = data  # fp is five-dimensional data, data is four-dimensional data
    end_toolbar()


if __name__ == '__main__':
    # 'argparse' module makes it easy to write a user-friendly command-line interface
    # that helps programmers define parameters for the model.
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='data/NTU-RGB-D/nturgb+d_skeletons_120')
    parser.add_argument(
        '--ignored_sample_path',
        default='resource/NTU-RGB-D/NTU_RGBD120_samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='data/NTU-RGB-D_120')

    benchmark = ['xsub', 'xview']  # Define the benchmark
    part = ['train', 'val']  # Define training set and test set
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)  # The output path is specified as xsub or xview, finally get .pkl and .npy files
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
# The .pkl and .npy files were obtained by processing the dataset files and can use for the following work