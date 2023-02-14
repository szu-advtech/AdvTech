import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Union

from pyntcloud import PyntCloud
import yaml

from utils import readply, writeply


def to_abs_path(path: Union[str, Path]) -> str:
    if isinstance(path, Path):
        return str(path.resolve())
    else:
        return str(Path(path).resolve())


def get_points_number(obj) -> int:
    if isinstance(obj, PyntCloud):
        return len(obj.points)
    if isinstance(obj, str):
        return get_points_number(Path(obj))
    if isinstance(obj, Path):
        if not obj.is_file():
            raise ValueError("{} is not a file".format(obj))
        point_cloud = PyntCloud.from_file(str(obj))
        return len(point_cloud.points)
    raise ValueError("type {} is not supported".format(type(obj)))


class GPCC:

    def __init__(self, exe_path: str, scale: float = 1, **kwds):
        exe_path = to_abs_path(exe_path)
        self.exe_path = exe_path if exe_path is not None else "tmc3"
        if shutil.which(self.exe_path) is None:
            raise RuntimeError("{} is not executable".format(self.exe_path))
        self.scale = scale

    def encode(
        self,
        inp: Union[str, List[str]],
        out: Union[str, List[str]],
        **kwds,
    ):
        """
        returns: {"out_size": int (B), "encode_time": float (s), "inp_bpp": float}
        """
        # {{{ check arguments
        # TODO: only support one file
        input_file = inp[0] if isinstance(inp, list) else inp
        output_file = out[0] if isinstance(out, list) else out
        input_file = to_abs_path(input_file)
        output_file = to_abs_path(output_file)
        # }}}

        # {{{ scale the ply
        # TODO do this in tester.py
        if not math.isclose(self.scale, 1):
            data = readply(input_file)
            data = data * self.scale
            scaled_ply_path = Path(output_file).parent / "{}.scaled.ply".format(
                Path(input_file).stem
            )
            writeply(str(scaled_ply_path), data)
            input_file = str(scaled_ply_path)
        # }}}

        return self.encode_octree(input_file, output_file, **kwds)

    def encode_octree(
        self, inp: str, out: str, lossless: bool = False, pqs: float = 1, **kwds
    ) -> dict:
        """
        pqs: positionQuantizationScale
        returns: a dict, {"out_size": int (B), "encode_time": float (s), "inp_bpp": float}
        """
        if not lossless and pqs is None:
            raise ValueError("when lossy encoding, pqs should be provided")
        parameters: dict = {}
        common_parameters = {
            "uncompressedDataPath": inp,
            "compressedStreamPath": out,
            # compress mode
            "mode": 0,
            # No attribute encoding
            "disableAttributeCoding": 0,
            # parameters appears in
            # mpeg-pcc-tmc13/cfg/octree-raht-ctc-*-geom-lossy-attrs.yaml
            "trisoupNodeSizeLog2": 0,
            "neighbourAvailBoundaryLog2": 8,
            "intra_pred_max_node_size_log2": 6,
            "maxNumQtBtBeforeOt": 4,
            "minQtbtSizeLog2": 0,
            "planarEnabled": 1,
            "planarModeIdcmUse": 0,
        }
        # parameters appears in
        # mpeg-pcc-tmc13/cfg/octree-raht-ctc-lossless-geom-lossy-attrs.yaml
        lossless_only_parameters = {
            "mergeDuplicatedPoints": 0,
            "positionQuantizationScale": 1,
            "inferredDirectCodingMode": 1,
        }
        # parameters appears in
        # mpeg-pcc-tmc13/cfg/octree-raht-ctc-lossy-geom-lossy-attrs.yaml
        lossy_only_parameters = {
            "mergeDuplicatedPoints": 1,
            "positionQuantizationScale": pqs,
        }
        # {**a, **b} to merge two dict
        if lossless:
            parameters = {**common_parameters, **lossless_only_parameters}
        else:
            parameters = {**common_parameters, **lossy_only_parameters}

        cli_args = self.generate_cli_args(self.exe_path, parameters)
        completed_process = subprocess.run(
            cli_args, capture_output=True, check=True, text=True, timeout=600
        )
        stdout: str = completed_process.stdout
        # collect result
        result = self.parse_tmc3_encode_output(stdout)
        result["inp_bpp"] = result["out_size"] * 8 / get_points_number(inp)
        return result

    def generate_cli_args(self, exe: str, par: dict):
        args = [exe]
        for key, value in par.items():
            arg = "--{}={}".format(key, value)
            args.append(arg)
        return args

    def parse_tmc3_encode_output(self, stdout: str) -> dict:
        """
        return: {"out_size": int (B), "encode_time": float (s)}
        tmc3 output example:
        ...
        Slice number: 1
        positions bitstream size 108856 B (1.01501 bpp)
        positions processing time (user): 3.004 s
        Total frame size 108892 B
        Total bitstream size 108892 B
        Processing time (wall): 5.469 s
        Processing time (user): 4.6 s
        """

        ret_dict = {}

        pattern = re.compile(r"Total bitstream size (\d+) B")
        search_res = pattern.search(stdout)
        if search_res is None:
            print(stdout.split("\n")[-10:])
            raise ValueError()
        bin_size = int(search_res.group(1))
        ret_dict["out_size"] = bin_size

        pattern = re.compile(r"Processing time \(wall\): ([\d.]+) s")
        search_res = pattern.search(stdout)
        if search_res is None:
            print(stdout.split("\n")[-10:])
            raise ValueError()
        encode_time = float(search_res.group(1))
        ret_dict["encode_time"] = encode_time

        return ret_dict

    def encode_trisoup(
        self,
        inp: str,
        out: str,
        node_size_log2: int,
        pqs: float = 1,
        **kwds,
    ):
        """
        geo_pre: geometry precision (bits)
        return: example: {"out_size": int (B), "encode_time": float (s), "inp_bpp": float}
        """
        # According to GPCC CTC, trisoupNodeSizeLog2 that controls the
        # compression quality has 4 ranks: R1 to R4, corresponding to
        # 5, 4, 3, 2
        if node_size_log2 not in {2, 3, 4, 5}:
            raise ValueError("quality should be one of {2, 3, 4, 5}")
        parameters = {
            "uncompressedDataPath": inp,
            "compressedStreamPath": out,
            # compress mode
            "mode": 0,
            # No attribute encoding
            "disableAttributeCoding": 0,
            # parameters appears in
            # mpeg-pcc-tmc13/cfg/trisoup-raht-ctc-lossy-geom-lossy-attrs.yaml
            "neighbourAvailBoundaryLog2": 8,
            "intra_pred_max_node_size_log2": 6,
            "inferredDirectCodingMode": 0,
            "planarEnabled": 1,
            "planarModeIdcmUse": 0,
            "positionQuantizationScale": pqs,
            "trisoupNodeSizeLog2": node_size_log2,
        }
        cli_args = self.generate_cli_args(self.exe_path, parameters)
        completed_process = subprocess.run(
            cli_args, capture_output=True, check=True, text=True
        )
        stdout: str = completed_process.stdout
        # collect result
        result = self.parse_tmc3_encode_output(stdout)
        result["inp_bpp"] = result["out_size"] * 8 / get_points_number(inp)
        return result

    def decode(self, inp, out, **kwds) -> dict:
        # TODO: only support one file
        input_file = inp[0] if isinstance(inp, list) else inp
        output_file = out[0] if isinstance(out, list) else out
        input_file = to_abs_path(input_file)
        output_file = to_abs_path(output_file)

        parameters = {
            "reconstructedDataPath": output_file,
            "compressedStreamPath": input_file,
            # uncompress mode
            "mode": 1,
            # ascii rather than binary
            "outputBinaryPly": 0,
            "convertPlyColourspace": 0,
        }
        cli_args = self.generate_cli_args(self.exe_path, parameters)
        completed_process = subprocess.run(
            cli_args, capture_output=True, check=True, text=True
        )
        stdout: str = completed_process.stdout

        # collect result
        result = self.parse_tmc3_decode_output(stdout)

        # {{{ scale the ply
        # TODO do this in tester.py
        if not math.isclose(self.scale, 1):
            # backup the original ply
            non_scaled_ply = str(
                Path(output_file).parent
                / "{}.original.ply".format(Path(output_file).stem)
            )
            shutil.move(output_file, non_scaled_ply)
            # do scaling
            data = readply(non_scaled_ply)
            data = data / self.scale
            writeply(output_file, data)
        # }}}

        return result

    def parse_tmc3_decode_output(self, stdout: str):
        """
        return: example: {"decode_time": float (s)}
        """

        """
        stdout example:
        positions bitstream size 108856 B
        positions processing time (user): 1.166 s
        Total bitstream size 108892 B
        Processing time (wall): 4.179 s
        Processing time (user): 1.283 s
        """
        ret_dict = {}

        pattern = re.compile(r"Processing time \(wall\): ([\d.]+) s")
        search_res = pattern.search(stdout)
        if search_res is None:
            print(stdout.split("\n")[-10:])
            raise ValueError("error")
        decode_time = float(search_res.group(1))
        ret_dict["decode_time"] = decode_time

        return ret_dict


def get_trisoup_pqs(res: float, geo_pre: float, test_depth: int = 9):
    if bool(geo_pre is None) != bool(geo_pre is None):
        raise ValueError("one and only one parameter should be provided")
    if geo_pre is None:
        # for example, resolution 1025 will be treated as 2^11
        geo_pre = math.ceil(math.log2(res))
    else:
        # in case geo_pre is a float
        geo_pre = math.ceil(geo_pre)
    # Scale to target geometry precision.
    # For example, to scale 2**11 to 2**9,
    # you have to multiply 2**(9-11) = 1/4.
    return 2 ** (test_depth - geo_pre)


def get_octree_lossy_pqs(res: float, geo_pre: float):
    """
    Give the set of positionQuantizationScale according to
    geometry precision.
    """
    d = {
        10: [15 / 16, 7 / 8, 3 / 4, 1 / 2, 1 / 4, 1 / 8],
        11: [7 / 8, 3 / 4, 1 / 2, 1 / 4, 1 / 8, 1 / 16],
        12: [3 / 4, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32],
        13: [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64],
        14: [1 / 4, 1 / 8, 1 / 16, 1 / 64, 1 / 128, 1 / 256],
        15: [1 / 4, 1 / 8, 1 / 32, 1 / 64, 1 / 256, 1 / 512],
    }
    if bool(geo_pre is None) != bool(geo_pre is None):
        raise ValueError("one and only one parameter should be provided")
    if geo_pre is None:
        # for example, resolution 1025 will be treated as 2^11
        geo_pre = math.ceil(math.log2(res))
    else:
        # in case geo_pre is a float
        geo_pre = math.ceil(geo_pre)
    min_key, max_key = min(d.keys()), max(d.keys())
    if geo_pre < min_key:
        return d[min_key]
    if geo_pre > max_key:
        return d[max_key]
    return d[geo_pre]

class CustomGPCC:
    
    @staticmethod
    def encode(
        inp: str,
        out: str,
        exe_path: str,
        cfg_file: str = None,
        extra_args: dict = None,
        drop_args: dict = None,
        **kwds,
    ):
        input_file = inp[0] if isinstance(inp, list) else inp
        input_file = to_abs_path(input_file)

        output_file = out[0] if isinstance(out, list) else out
        output_file = to_abs_path(output_file)

        d = {
            "uncompressedDataPath": input_file,
            "compressedStreamPath": output_file,
            # compress mode
            "mode": 0,
        }
        if cfg_file:
            cfg = CustomGPCC.load_cfg(cfg_file)
            d.update(cfg)
        # add args
        if extra_args:
            d.update(extra_args)
        # drop args
        if drop_args:
            for drop_key in drop_args:
                d.pop(drop_key)

        # dict -> CLI arg string
        cli_args: list[str] = CustomGPCC.dict_to_cli_args(d)
        cli_args = [to_abs_path(exe_path)] + cli_args
        # Run
        try:
            completed_process = subprocess.run(
                cli_args, capture_output=True, check=True, text=True, timeout=600
            )
        except subprocess.CalledProcessError as e:
            print(cli_args)
            print(f"Error on CMD: {e.cmd}")
            print(f"Ouput: {e.output}")
            print(f"Return Code: {e.returncode}")
            return
        stdout: str = completed_process.stdout
        # Result
        result = CustomGPCC.parse_tmc3_output(stdout)
        return result

    @staticmethod
    def decode(
        inp: str,
        out: str,
        exe_path: str,
        cfg_file: str = None,
        extra_args: dict = None,
        drop_args: dict = None,
        **kwds,
    ):
        input_file = inp[0] if isinstance(inp, list) else inp
        input_file = to_abs_path(input_file)

        output_file = out[0] if isinstance(out, list) else out
        output_file = to_abs_path(output_file)

        d = {
            "compressedStreamPath": input_file,
            "reconstructedDataPath": output_file,
            # uncompress mode
            "mode": 1,
            # ascii rather than binary
            "outputBinaryPly": 0,
        }
        if cfg_file:
            cfg = CustomGPCC.load_cfg(cfg_file)
            d.update(cfg)
        # add args
        if extra_args:
            d.update(extra_args)
        # drop args
        if drop_args:
            for drop_key in drop_args:
                d.pop(drop_key)

        # dict -> CLI arg string
        cli_args: list[str] = CustomGPCC.dict_to_cli_args(d)
        cli_args = [to_abs_path(exe_path)] + cli_args
        # Run
        try:
            completed_process = subprocess.run(
                cli_args, capture_output=True, check=True, text=True, timeout=600
            )
        except subprocess.CalledProcessError as e:
            print(f"Error on CMD: {e.cmd}")
            print(f"Ouput: {e.output}")
            print(f"Return Code: {e.returncode}")
            return
        stdout: str = completed_process.stdout
        # Result
        result = CustomGPCC.parse_tmc3_output(stdout)
        return result

    @staticmethod
    def load_cfg(cfg_file: str):
        with open(cfg_file, "r") as f:
            lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            # "a: 1, 2, 3, 4\n" -> "a: [1, 2, 3, 4]\n"
            if "," in line:
                # key = "a", value = "1, 2, 3, 4\n"
                key, value = line.split(": ")
                lines[i] = key + ": " + "[" + value[:-1] + "]" + "\n"
        cfg = yaml.safe_load("".join(lines))
        # cfg = [yaml.safe_load(line) for line in lines]
        # There may be repeat keys in cfg file,
        # the later should overwrites the former, yaml will do it auto
        return cfg

    @staticmethod
    def dict_to_cli_args(cfg: dict) -> List[str]:
        # Convert to str
        d = {}
        for key, value in cfg.items():
            new_value = value
            if isinstance(new_value, list):
                # TODO nested list
                # [1.0, "a"] ->"1.0, a"
                new_value = ", ".join(map(str, new_value))
            else:
                new_value = str(new_value)
            d[str(key)] = new_value
        # To list of strings
        cli_args = []
        for key, value in d.items():
            arg = f"--{key}={value}"
            cli_args.append(arg)
        return cli_args

    @staticmethod
    def parse_tmc3_output(stdout: str) -> dict:
        ret_dict = {}

        search = [
            # time
            (r"Processing time \(wall\): ([\d.]+) s", "time", 1, float),
            # bit size
            (r"Total bitstream size (\d+) B", "total_size", 1, int),
            (r"positions bitstream size (\d+) B", "geom_size", 1, int),
            (r"colors bitstream size (\d+) B", "color_size", 1, int),
            (r"reflectances bitstream size (\d+) B", "refl_size", 1, int),
            # bpp
            (
                r"positions bitstream size (\d+) B \(([\d.]+) bpp\)",
                "geom_bpp",
                2,
                float,
            ),
            (r"colors bitstream size (\d+) B \(([\d.]+) bpp\)", "color_bpp", 2, float),
            (
                r"reflectances bitstream size (\d+) B \(([\d.]+) bpp\)",
                "refl_bpp",
                2,
                float,
            ),
        ]

        for re_string, save_name, search_pos, trans_fun in search:
            pattern = re.compile(re_string)
            search_res = pattern.search(stdout)
            if search_res:
                ret_dict[save_name] = trans_fun(search_res.group(search_pos))

        return ret_dict
