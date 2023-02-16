import os
import time
import numpy as np
import pandas as pd
import time
from pyntcloud import PyntCloud

from config import config


class Rotation:
    """
    Class that describes the rotation of user's FoV, measured in rad.
    """

    def __init__(self, rot: tuple) -> None:
        self.x, self.y, self.z = rot

        rotX = np.asarray(
            [
                [1, 0, 0, 0],
                [0, np.cos(self.x), np.sin(self.x), 0],
                [0, -np.sin(self.x), np.cos(self.x), 0],
                [0, 0, 0, 1],
            ]
        )

        rotY = np.asarray(
            [
                [np.cos(self.y), 0, np.sin(self.y), 0],
                [0, 1, 0, 0],
                [-np.sin(self.y), 0, np.cos(self.y), 0],
                [0, 0, 0, 1],
            ]
        )

        rotZ = np.asarray(
            [
                [np.cos(self.z), np.sin(self.z), 0, 0],
                [-np.sin(self.z), np.cos(self.z), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        self.matrix = rotX @ rotY @ rotZ


class Translation:
    """
    Class that describes the translation of user's FoV.
    """

    def __init__(self, trans: tuple) -> None:
        self.x, self.y, self.z = trans

        self.matrix = np.asarray(
            [
                [1, 0, 0, self.x],
                [0, 1, 0, self.y],
                [0, 0, 1, self.z],
                [0, 0, 0, 1],
            ]
        )


class Movement:
    """
    Class that describes user's FoV.
    """

    def __init__(self, rotation: Rotation, translation: Translation) -> None:
        self.matrix = rotation.matrix @ translation.matrix


class ViewFrustum:

    def __init__(self, *shape) -> None:
        r, t, n, f = shape
        self.x_upper = lambda z: z * r / -f
        self.x_lower = lambda z: z * -r / -f
        self.y_upper = lambda z: z * t / -f
        self.y_lower = lambda z: z * -t / -f
        self.z_upper = lambda z: -n
        self.z_lower = lambda z: -f


class User:
    """
    Class that describes a user following a certain server.
    """

    def __init__(self, config: dict) -> None:
        self.movement_sequence = [
            Movement(
                rotation=Rotation(m["rotation"]),
                translation=Translation(m["translation"]),
            )
            for m in config["movement_path"]
        ]
        self.movement_index = 0
        self.modelview = np.eye(4, 4)

        r, t, n, f = config["view_frustum"]
        self.viewfrustum = ViewFrustum(r, t, n, f)
        self.projection = np.asarray(
            [
                [n / r, 0, 0, 0],
                [0, n / t, 0, 0],
                [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
                [0, 0, -1, 0],
            ]
        )

        w, h, n, f, x, y = config["view_screen"]
        self.viewport = np.asarray(
            [
                [w / 2, 0, 0, x + w / 2],
                [0, h / 2, 0, y + h / 2],
                [0, 0, (f - n) / 2, (n + f) / 2],
                [0, 0, 0, 1],
            ]
        )

    def _update_modelview(self) -> None:
        self.modelview = self.movement_sequence[self.movement_index].matrix
        self.movement_index += 1

    def movement(self) -> Movement:
        return self.movement_sequence[self.movement_index]

    def is_collided(self, tile_centroids: np.ndarray) -> np.ndarray:
        # assume that the `tile_centroids` was in such a format:
        # shape: [N, 4]
        # entry: [x, y, z, 1]
        tile_centroids = (self.modelview @ tile_centroids.T).T
        upper = np.ones_like(tile_centroids)
        lower = np.ones_like(tile_centroids)

        upper[:, 0] = self.viewfrustum.x_upper(tile_centroids[:, 2])
        upper[:, 1] = self.viewfrustum.y_upper(tile_centroids[:, 2])
        upper[:, 2] = self.viewfrustum.z_upper(tile_centroids[:, 2])

        lower[:, 0] = self.viewfrustum.x_lower(tile_centroids[:, 2])
        lower[:, 1] = self.viewfrustum.y_lower(tile_centroids[:, 2])
        lower[:, 2] = self.viewfrustum.z_lower(tile_centroids[:, 2])

        return np.logical_and(lower <= tile_centroids, tile_centroids <= upper).all(axis=1)

    def visable_points(self, pointcloud: np.ndarray) -> np.ndarray:
        # assume that the `pointcloud` was in such a format:
        # shape: [N, 5]
        # entry: [x, y, z, 1, [attrs], i]
        tik = time.time()
        pointcloud, attrs, indices = pointcloud[:, :4], pointcloud[:, 4:-1], pointcloud[:, -1]
        pointcloud = pointcloud.T
        pointcloud = self.projection @ self.modelview @ pointcloud

        # record `z`s before clipped for overlapped points selection
        depth = pointcloud[2, :]
        pointcloud = pointcloud / pointcloud[-1, :]
        # remove all points that out of range
        pointcloud = pointcloud.T
        pointcloud = np.concatenate(
            [
                pointcloud,
                attrs,
                np.expand_dims(depth, axis=1),
                np.expand_dims(indices, axis=1),
            ], axis=1)
        pointcloud = pointcloud[
            (pointcloud[:, 0] >= -1) & (pointcloud[:, 0] <= 1) & 
            (pointcloud[:, 1] >= -1) & (pointcloud[:, 1] <= 1) &
            (pointcloud[:, 2] >= -1) & (pointcloud[:, 2] <= 1), :]
        pointcloud, attrs, depth, indices = \
            pointcloud[:, :4], pointcloud[:, 4:-2], pointcloud[:, -2], pointcloud[:, -1]
        pointcloud = pointcloud.T

        pointcloud = self.viewport @ pointcloud
        pointcloud = pointcloud.T
        pointcloud = np.concatenate(
            [
                pointcloud,
                attrs,
                np.expand_dims(depth, axis=1),
                np.expand_dims(indices, axis=1),
            ], axis=1)
        print(f"求取点屏幕坐标耗时：{time.time()-tik}秒")
        # assume that the `pointcloud` was in such a format:
        # shape: [N, 6]
        # entry: [x, y, z, 1, [attrs], depth, i]
        tik = time.time()
        pointcloud = pointcloud[pointcloud[:, -2].argsort()]
        pointcloud = pointcloud[pointcloud[:, 1].argsort()]
        pointcloud = pointcloud[pointcloud[:, 0].argsort()]
        delta = np.zeros_like(pointcloud)
        delta[:-1, :] = pointcloud[1:, :]
        delta_pointcloud = np.round(pointcloud) - np.round(delta)
        vis_points = pointcloud[
            (delta_pointcloud[:, 0] != 0) | 
            (delta_pointcloud[:, 1] != 0), :
        ]
        vis_points = np.delete(vis_points, -2, axis=1)
        print(f"判断遮挡关系、求取可见点集耗时：{time.time()-tik}秒")

        return vis_points


class Server:
    """
    Class that describes a server.
    """

    def __init__(self, config: dict) -> None:
        self.user = User(config["user"])
        self.BandWidth = config["BandWidth"]

    def update_frame(self, PtClPath: str):
        self.PtCl_Path = PtClPath
        self.D1toD3 = np.load(os.path.join(self.PtCl_Path, "1Dto3D.npy"))
        self.BitNum = np.load(os.path.join(self.PtCl_Path, "BitNum.npy"))
        self.centroids = np.load(os.path.join(self.PtCl_Path, "centroids.npy"))

        self.user._update_modelview()

    def run(self):
        tik = time.time()
        vis_tiles = self.user.is_collided(
            np.insert(self.centroids, 3, values=1, axis=1))
        print(f"判断可见瓦片耗时：{time.time()-tik}秒")
        pointcloud = []
        points_of_tile = {}
        tik = time.time()
        for idx in range(vis_tiles.shape[0]):
            if vis_tiles[idx]:
                i, j, k = self.D1toD3[idx]
                tile = readply(os.path.join(
                    self.PtCl_Path, "ply", f"{i}_{j}_{k}.ply"))
                tile = np.insert(tile, tile.shape[1], values=idx, axis=1)
                pointcloud.append(tile)
                points_of_tile[idx] = tile.shape[0]
        print(f"读取可见瓦片文件耗时：{time.time()-tik}秒")
        print("#tiles: {}".format(len(pointcloud)))
        pointcloud = np.concatenate(pointcloud, axis=0)
        pointcloud = np.insert(pointcloud, 3, values=1, axis=1)

        vis_points = self.user.visable_points(pointcloud)
        tik = time.time()
        vis_points_of_tile = {}
        for idx in range(vis_points.shape[0]):
            if vis_points[idx, -1] not in vis_points_of_tile.keys():
                vis_points_of_tile[vis_points[idx, -1]] = 1
            else:
                vis_points_of_tile[vis_points[idx, -1]] += 1

        vis_to_idx = [int(idx) for idx in vis_points_of_tile.keys()]
        MatrixB = np.zeros([len(vis_points_of_tile), 6])
        MatrixQ = np.zeros([len(vis_points_of_tile), 6])
        for vis_idx in range(MatrixB.shape[0]):
            idx = vis_to_idx[vis_idx]
            MatrixB[vis_idx, :] = self.BitNum[idx, :]
            MatrixQ[vis_idx, :] = \
                vis_points_of_tile[idx] / points_of_tile[idx] * \
                self.BitNum[idx, :] / self.BitNum[idx, -1]
        DeltaB = (MatrixB.T - MatrixB[:, 0]).T[:, 1:]
        DeltaQ = (MatrixQ.T - MatrixQ[:, 0]).T[:, 1:]
        B1 = np.sum(MatrixB[:, 0])
        Q1 = np.sum(MatrixQ[:, 0])
        print(f"生成所需矩阵数据耗时：{time.time()-tik}秒")

        tik = time.time()
        Acb = np.zeros([len(vis_points_of_tile), 5])
        while True:
            X = -np.inf * np.ones_like(Acb)
            DeltaB_of_AcbUx = (DeltaB * Acb).sum() + DeltaB
            DeltaQ_of_AcbUx = ((DeltaQ * Acb).sum() + DeltaQ.T).T
            # DeltaQ_of_AcbUx = ((DeltaQ * Acb).sum() - (DeltaQ * Acb).sum(axis=1) + DeltaQ.T).T
            is_AcbUx = (Acb != 1)
            is_BandWidth_allowed = (DeltaB_of_AcbUx <= (self.BandWidth - B1))
            if not (is_AcbUx & is_BandWidth_allowed).any():
                break
            X = np.where(
                is_AcbUx & is_BandWidth_allowed,
                (DeltaQ_of_AcbUx - (DeltaQ * Acb).sum()) / (DeltaB + 1e-6),
                X,
            )
            i, q = np.unravel_index(X.argmax(), X.shape)
            Acb[i, q] = 1
            # X = []
            # for i in range(Acb.shape[0]):
            #     for q in range(1, Acb.shape[1]):
            #         # if not Acb[i, q:].sum() >= 1:
            #         if not Acb[i, q] == 1:
            #             AcbUx = Acb.copy()
            #             AcbUx[i, q] = 1
            #             AcbUx[i, :q] = 0
            #             if np.sum(DeltaB * AcbUx) <= self.BandWidth - B1:
            #                 X.append([
            #                     i,
            #                     q,
            #                     (np.sum(DeltaQ * AcbUx) - np.sum(DeltaQ * Acb)) /
            #                     (DeltaB[i, q] + 1e-6)
            #                 ])
            # if len(X) == 0:
            #     break
            # i, q, _ = max(X, key=lambda x: x[2])
            # Acb[i, q] = 1
            # Acb[i, :q] = 0
            # print("\r#BitAcb: ", np.sum(DeltaB * Acb) + B1, end="")
        print("#BitAcb: ", np.sum(DeltaB * Acb) + B1)
        print("QoE Acb: ", np.sum(DeltaQ * Acb) + Q1)
        # print()
        print(f"求解Acb耗时：{time.time()-tik}秒")

        Aub = np.zeros([len(vis_points_of_tile), 5])
        tik = time.time()
        while True:
            X = -np.inf * np.ones_like(Aub)
            DeltaB_of_AubUx = (DeltaB * Aub).sum() + DeltaB
            DeltaQ_of_AubUx = ((DeltaQ * Aub).sum() + DeltaQ.T).T
            # DeltaQ_of_AubUx = ((DeltaQ * Aub).sum() - (DeltaQ * Aub).sum(axis=1) + DeltaQ.T).T
            is_AubUx = (Aub != 1)
            is_BandWidth_allowed = (DeltaB_of_AubUx <= (self.BandWidth - B1))
            if not (is_AubUx & is_BandWidth_allowed).any():
                break
            X = np.where(
                is_AubUx & is_BandWidth_allowed,
                DeltaQ_of_AubUx - (DeltaQ * Aub).sum(),
                X,
            )
            i, q = np.unravel_index(X.argmax(), X.shape)
            Aub[i, q] = 1
            # X = []
            # for i in range(Aub.shape[0]):
            #     for q in range(1, Aub.shape[1]):
            #         # if not Aub[i, q:].sum() >= 1:
            #         if not Aub[i, q] == 1:
            #             AubUx = Aub.copy()
            #             AubUx[i, q] = 1
            #             AubUx[i, :q] = 0
            #             if np.sum(DeltaB * AubUx) <= self.BandWidth - B1:
            #                 X.append([
            #                     i,
            #                     q,
            #                     np.sum(DeltaQ * AubUx) - np.sum(DeltaQ * Aub)
            #                 ])
            # if len(X) == 0:
            #     break
            # i, q, _ = max(X, key=lambda x: x[2])
            # Aub[i, q] = 1
            # Aub[i, :q] = 0
            # print("\r#BitAub: ", np.sum(DeltaB * Aub) + B1, end="")
        print("#BitAub: ", np.sum(DeltaB * Aub) + B1)
        print("QoE Aub: ", np.sum(DeltaQ * Aub) + Q1)
        print(f"求解Aub首次迭代耗时：{time.time()-tik}秒")
        # print()

        A = np.zeros([len(vis_points_of_tile), 6])
        A[:, 1:] = Acb if np.sum(DeltaQ * Acb) > np.sum(DeltaQ * Aub) else Aub
        solution = np.zeros([vis_tiles.shape[0], 6])
        for vis_idx in range(A.shape[0]):
            if np.sum(A[vis_idx, :]) == 0:
                A[vis_idx, 0] = 1
            idx = vis_to_idx[vis_idx]
            solution[idx, :] = A[vis_idx, :]
        
        return solution, (A * MatrixQ).sum(), vis_points


def readply(filepath) -> np.ndarray:
    attrs=["x", "y", "z", "red", "green", "blue", "alpha"]
    point_cloud = PyntCloud.from_file(filepath)
    try:
        points_df = point_cloud.points[attrs]
    except:
        point_cloud.points["alpha"] = 255
        points_df = point_cloud.points[attrs]
    point_cloud = points_df.to_numpy()
    return point_cloud


def writeply(filepath, points) -> None:
    attrs=["x", "y", "z", "red", "green", "blue", "alpha"]
    points_df = pd.DataFrame(points, columns=attrs)
    points_df[attrs[-4:]] = points_df[attrs[-4:]].astype(np.uint8)
    point_cloud = PyntCloud(points_df)
    return point_cloud.to_file(filepath, as_text=True)


if __name__ == "__main__":
    server = Server(config["server"])
    server.user._update_modelview()
    solution = server.run()
    np.savetxt("./sln.txt", solution)
