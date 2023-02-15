import os
import sys
import time
import subprocess

if __name__ == "__main__":
    # # detection
    # subprocess.call("CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 4 0 0 &"
    #                 # "CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 7 1 0 &"
    #                 # "CUDA_VISIBLE_DEVICES=2 python -u ./table_recognition/table_inference.py 7 2 0 &"
    #                 # "CUDA_VISIBLE_DEVICES=3 python -u ./table_recognition/table_inference.py 7 3 0 &"
    #                 "CUDA_VISIBLE_DEVICES=4 python -u ./table_recognition/table_inference.py 4 1 0 &"
    #                 "CUDA_VISIBLE_DEVICES=5 python -u ./table_recognition/table_inference.py 4 2 0 &"
    #                 # "CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 8 6 0 &"
    #                 "CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 4 3 0", shell=True)
    # time.sleep(60)

    # # structure
    # subprocess.call("CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 4 0 2 &"
    #                 # "CUDA_VISIBLE_DEVICES=1 python -u ./table_recognition/table_inference.py 7 1 2 &"
    #                 # "CUDA_VISIBLE_DEVICES=2 python -u ./table_recognition/table_inference.py 7 2 2 &"
    #                 # "CUDA_VISIBLE_DEVICES=3 python -u ./table_recognition/table_inference.py 7 3 2 &"
    #                 "CUDA_VISIBLE_DEVICES=4 python -u ./table_recognition/table_inference.py 4 1 2 &"
    #                 "CUDA_VISIBLE_DEVICES=5 python -u ./table_recognition/table_inference.py 4 2 2 &"
    #                 # "CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 8 6 2 &"
    #                 "CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 4 3 2", shell=True)
    # time.sleep(60)

    # recognition
    subprocess.call("CUDA_VISIBLE_DEVICES=6 python -u ./table_recognition/table_inference.py 1 0 1", shell=True)