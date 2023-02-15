import numpy as np
import os


def read_skeleton(file):  # Pass in the 'skeleton' file
    with open(file, 'r') as f:
        skeleton_sequence = {}  # Dictionary type, and is the final return result
        """
        Line 1: Number of frames. The number 71 indicates that the 'skeleton' file has 71 frames;

        The second line: the number of bodies. The number 1 indicates that a body appears in this video frame;

        Line 3: There are 10 numbers, in order: 'bodyID', 'clipedEdges', 'handLeftConfidence',
        'handLeftState', 'handRightConfidence', 'handRightState', 'isResticted', 'leanX', 'leanY', 'trackingState';

        Line 4: number of joints. The number 25 means there are 25 joints;

        Lines 5-29: Data for 25 joints with 12 numbers, in order: 'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
         'orientationW','orientationX', 'orientationY', 'orientationZ', 'trackingState';

        Then there is the data for frame NO.2, following the above rules.

        """
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []  # Record the specific data of each frame
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))  # Number of joint features, number of frames, number of joints, number of people in a frame
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data  # return a 4-dimensional numpy matrix