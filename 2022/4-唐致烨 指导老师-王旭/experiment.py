import importlib
import os
import shutil
import time
from collections import defaultdict
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np

from config import config
from gpcc import CustomGPCC as GPCC
from utils import *


class Video:

    def __init__(self, data: dict):

        self.videoname = data["videoname"]
        self.filename = data["filename"]
        self.frames = data["frames"]
        self.multiprocessing = data["multiprocessing"]

        self.codec_path = "/home/zhiyetang/data/codes/mpeg-pcc-tmc13/build/tmc3/tmc3"
        self.encoder_cfg = lambda q: f"videos/cfgs/r{str(q+1).zfill(2)}/encoder.cfg"
        self.decoder_cfg = lambda q: f"videos/cfgs/r{str(q+1).zfill(2)}/decoder.cfg"
        self.SPLITSTEP = 30

    def process(self, codec: GPCC):
        # implement multi-processings
        process_list = []
        frames_per_process = int(np.ceil(self.frames / self.multiprocessing))
        for tid in range(1, self.multiprocessing+1):
            p = Process(target=self._process,
                        args=(tid, (tid-1)*frames_per_process+1, tid*frames_per_process, codec))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

    def _process(self, thread_id: int, bgn_frame: int, end_frame: int, codec: GPCC):
        bgn_frame = max(1, bgn_frame)
        end_frame = min(end_frame, self.frames)
        for f in range(bgn_frame, end_frame+1):
            name = os.path.join("videos", self.videoname, self.filename(f))
            outname = os.path.join(
                "videos", self.videoname, self.filename(f)[:-4])

            if not os.path.exists(outname):
                print("TID {}: Processing Frame No {}/{}".
                      format(str(thread_id).zfill(3), str(f).zfill(8), str(end_frame).zfill(8)))
                os.makedirs(outname, exist_ok=True)
                pointcloud = readply(name)
                self._frame_process(codec, pointcloud, outname)
                shutil.move(name, outname)

    def _frame_process(
        self,
        codec: GPCC,
        pointcloud: np.ndarray,
        outname: str
    ):
        # tile up
        indices = np.floor(pointcloud[:, :3] / self.SPLITSTEP).astype(np.int32)
        d = defaultdict(list)
        for point, index in zip(pointcloud.tolist(), indices.tolist()):
            d[tuple(index)].append(point)
        dic = []
        # save .ply and .bin
        os.makedirs(os.path.join(outname, "ply"), exist_ok=True)
        os.makedirs(os.path.join(outname, "bin"), exist_ok=True)
        os.makedirs(os.path.join(outname, "reconstructed"), exist_ok=True)
        for index, point_list in list(d.items()):
            dic.append(index)
            i, j, k = index
            out_ply_name = os.path.join(outname, "ply", f"{i}_{j}_{k}.ply")
            writeply(out_ply_name, np.array(point_list))
            for q in range(6):
                out_bin_name = os.path.join(
                    outname, "bin", f"{i}_{j}_{k}_{q}.bin")
                rec_ply_name = os.path.join(
                    outname, "reconstructed", f"{i}_{j}_{k}_{q}.ply")
                codec.encode(
                    inp=out_ply_name,
                    out=out_bin_name,
                    exe_path=self.codec_path,
                    cfg_file=self.encoder_cfg(q),
                )
                codec.decode(
                    inp=out_bin_name,
                    out=rec_ply_name,
                    exe_path=self.codec_path,
                    cfg_file=self.decoder_cfg(q)
                )

        centroids = []
        dic = np.array(dic).astype(np.int32)
        np.save(os.path.join(outname, "1Dto3D.npy"), dic)
        I, J, K = dic.max(axis=0)
        idic = np.zeros([I+1, J+1, K+1]).astype(np.int32)
        BitNum = np.zeros([dic.shape[0], 6])
        for idx in range(dic.shape[0]):
            i, j, k = dic[idx, :]
            centroids.append([
                (i + .5) * self.SPLITSTEP,
                (j + .5) * self.SPLITSTEP,
                (k + .5) * self.SPLITSTEP,
            ])
            idic[i, j, k] = idx
            for q in range(6):
                bin_path = os.path.join(outname, "bin", f"{i}_{j}_{k}_{q}.bin")
                BitNum[idx, q] = os.path.getsize(bin_path) * 1.

        np.save(os.path.join(outname, "3Dto1D.npy"), idic)
        np.save(os.path.join(outname, "centroids.npy"), np.array(centroids))
        np.save(os.path.join(outname, "BitNum.npy"), BitNum)


class Runner:

    def __init__(self, config):

        self.gpcc = GPCC()
        self.videodata = importlib.import_module(
            "videos." + config["video"] + ".data").data
        self.video = Video(self.videodata)
        self.video.process(self.gpcc)

        self.server = Server(config["server"])

    def run(self):
        for f in range(350, self.video.frames+1):
            self.server.update_frame(os.path.join(
                "videos", self.videodata["videoname"], self.video.filename(f)[:-4]))
            scheme_tik = time.time()
            sln, qoe, pts = self.server.run()
            scheme_tok = time.time()
            print(f"Scheme Time Cost: {scheme_tok - scheme_tik}")

            rec_tik = time.time()
            reconstructed = []
            for idx in range(sln.shape[0]):
                q = np.argwhere(sln[idx, :] == 1)
                if q.size > 0:
                    i, j, k = self.server.D1toD3[idx]
                    q = int(q.max())
                    reconstructed.append(readply(os.path.join(
                        "videos",
                        self.videodata["videoname"],
                        self.video.filename(f)[:-4],
                        "reconstructed",
                        f"{i}_{j}_{k}_{q}.ply"
                    )))
            reconstructed = np.concatenate(reconstructed, axis=0)
            writeply("out.ply", reconstructed)
            reconstructed = np.insert(reconstructed, 3, values=1, axis=1)
            reconstructed = np.insert(reconstructed, 8, values=0, axis=1)
            reconstructed = self.server.user.visable_points(reconstructed)
            rec_tok = time.time()
            print(f"Reconstruction Time Cost: {rec_tok - rec_tik}")

            render_tik = time.time()
            plt.scatter(
                reconstructed[:, 0], reconstructed[:, 1], s=1,
                c=reconstructed[:, 4:8]/255,
            )
            plt.axis("equal")
            plt.savefig(f"./output/{str(f).zfill(8)}.png")
            plt.clf()
            render_tok = time.time()
            print(f"Rendering Time Cost: {render_tok - render_tik}")

            print(f"Frame No {str(f).zfill(8)}: {qoe}\n")


if __name__ == "__main__":
    runner = Runner(config)
    runner.run()
