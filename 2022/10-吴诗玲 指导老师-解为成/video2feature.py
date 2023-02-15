from pathlib import Path
import numpy as np
import h5py

from load_dll import GetFeature

def main(args):
# def main():
    video_path = Path("/NAS_REMOTE/zhouyz/dataset/depression/part2/videos")
    target_path = Path("feature")
    meh = GetFeature(args.path[0])
    # meh = GetFeature("./sharelib/libOpenFaceWrapper.so")
    fname = str(target_path / "part2_1.hdf5")
    num_video = len(video_path.iterdir())
    with h5py.File(fname, "w") as f:
        for i, video in enumerate(video_path.iterdir()):
            feature = meh.get_FeaWithoutPostProcess(str(video))#获取处理后的特征
            f.create_dataset(video.stem, data=feature)
            print(f"{i}/{num_video}")

    


if __name__== "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = 'OpenFaceWrapper shared lib test.')
    parser.add_argument('path', nargs=1, help='Path to OpenFace shared lib.')
    args = parser.parse_args()
    main(args)
    # main()
    # with h5py.File("feature/part2_1.hdf5", "r") as f:
    #     print(f.filename, ":")
    #     for k, v in f.items():
    #         print(k, v.shape)