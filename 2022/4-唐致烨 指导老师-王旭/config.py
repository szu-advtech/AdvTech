import json
import numpy as np

config = {
    "tmc_path": "/home/zhiyetang/data/codes/mpeg-pcc-tmc13/build/tmc3/tmc3",
    "video": "basketball_player_vox11",
    "server": {
        "user": {
            "movement_path": json.load(
                open("./movement_path/fix.json", "r")
            ),
            "view_frustum": [1600, 1600, 1600, 4800],
            "view_screen": [1600, 1600, 0.1, 4800, 0, 0]
        },
        "BandWidth": 1000000
    }
}