import platform
if platform.system() == "Windows":
    # tk can't be imported on the server right now : (
    import tkinter as tk
    from tkinter import filedialog

import os
import abs_utils
import ast
import os
import sys
import rl_config
import dataset
import numpy as np
from global_config import ss_len


def write_abs_prcs(abstraction_process):
    abs_dir = rl_config.abstraction_process_log_dir()

    for episode in abstraction_process.keys():
        ep_dir: str = abs_dir + "/" + str(episode)
        os.makedirs(ep_dir, exist_ok=True)

        ep_abs_prcs = abstraction_process[episode]
        diff_idx = 0  # differentiate sketches in one ep
        for abs_prcs in ep_abs_prcs:
            s_data = abs_prcs[0]
            r_sidx = abs_prcs[1]
            s_np = s_data[0]
            label_gt = s_data[1]
            abs_file = ep_dir + "/" + dataset.labels[label_gt] + "_{}.txt".format(diff_idx)
            diff_idx += 1
            np.set_printoptions(threshold=sys.maxsize)
            with open(abs_file, "w") as f:
                f.write("{}\n\n{}".format(np.array2string(s_np), str(r_sidx)))
    return


def __parse_sketch_np_str__(sketch_str: str, points_num: int):
    sketch_np = np.zeros(shape=(points_num, 3))
    point_str_list = sketch_str.split("\n")
    for r in range(points_num):
        point_str = point_str_list[r]
        # point value: ['-0.01273885', '-0.30980393', '', '0.', '', '', '', '', '', '', '', '']
        values = point_str.replace("[", "").replace("]", "").split(" ")
        i: int = 0
        j: int = 0
        while i < 3 and j < len(values):
            if values[j] != '':
                sketch_np[r][i] = float(values[j])
                i += 1
            j += 1

    return sketch_np


def read_sketch_abs_prcs(sketch_abs_filepath: str):
    if len(sketch_abs_filepath) < 1:
        raise AssertionError("sketch_abs_filepath is empty string")

    sketch_str = ""
    points_num: int = 0
    removed_strokes_idx = None
    with open(sketch_abs_filepath, "r") as f:
        # read sketch_np string
        while True:
            line: str = f.readline()
            if line == '\n':
                break
            sketch_str = sketch_str + line
            points_num += 1

        # read removed strokes idx
        line: str = f.readline()
        removed_strokes_idx = ast.literal_eval(line)

    sketch_np = __parse_sketch_np_str__(sketch_str, points_num)
    return sketch_np, removed_strokes_idx


def show_abs_prcs():
    root = tk.Tk()
    root.withdraw()

    while True:
        file_path: str = filedialog.askopenfilename(title="select", initialdir=os.getcwd() + "/abstraction_process")
        if file_path == "":
            break

        print(file_path)
        sketch_np, rm_ss_indices = read_sketch_abs_prcs(file_path)
        print(sketch_np)
        print("removed_stroke_seg_indices: {}".format(rm_ss_indices))
        abs_utils.plot_sketch(sketch_data=sketch_np, end_on_removed=False)

        for rm_ss_idx in rm_ss_indices:
            start: int = rm_ss_idx * ss_len()
            end: int = len(sketch_np) \
                if (rm_ss_idx + 1) * ss_len() >= len(sketch_np) \
                else (rm_ss_idx + 1) * ss_len()


            # remove current stroke
            # end: int = len(sketch_np)
            # if i + 1 < len(removed_strokes_idx):
            #     end = stroke_offset[i + 1]
            delta_x, delta_y = 0.0, 0.0
            for j in range(start, end):
                point = sketch_np[j]
                j += 1  # reference to next point

                delta_x += point[0]
                delta_y += point[1]
                point[0] = point[1] = 0.0

            last_endpoint_idx = start - 1
            while last_endpoint_idx >= 0:
                p = sketch_np[last_endpoint_idx]
                if p[0] != 0.0 and p[1] != 0.0:
                    break
                last_endpoint_idx -= 1

            if last_endpoint_idx >= 0:
                p = sketch_np[last_endpoint_idx]
                p[2] = 1.0

            if end < len(sketch_np):
                sketch_np[end][0] += delta_x
                sketch_np[end][1] += delta_y

            print(sketch_np)

            abs_utils.plot_sketch(sketch_data=sketch_np, end_on_removed=False)
    return


if __name__ == "__main__":
    show_abs_prcs()
