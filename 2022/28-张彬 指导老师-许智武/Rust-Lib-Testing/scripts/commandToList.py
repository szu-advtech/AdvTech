#!/usr/bin/python3

import sys
import json

if __name__ == '__main__':
    command = sys.argv[1:]
    print(json.dumps(command))
    