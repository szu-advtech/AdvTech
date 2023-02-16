#!/bin/bash

python3 ../mtool/main.py $2--strings --ids --read mrp --write norec "$1" "$1_converted"
