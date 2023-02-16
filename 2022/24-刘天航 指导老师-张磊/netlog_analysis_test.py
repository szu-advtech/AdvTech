import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import random

from matplotlib.ticker import StrMethodFormatter
from glob import glob
from pathlib import Path
from collections import defaultdict, deque
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlog")
    parser.add_argument("--host")
    parser.add_argument("--title")

    args = parser.parse_args()

    print(args)
    
if __name__ == "__main__":
    main()