#!/usr/bin/env bash

DIR_BACKUP=$(pwd)
if [ $BASH_SOURCE ];then
    cd $(dirname $BASH_SOURCE)
    ROOT_DIR="$(dirname `pwd`)"
else
    ROOT_DIR=$(pwd)
fi
cd $DIR_BACKUP

read -n 1 -p "Is this your tool's root directory: $ROOT_DIR? (Y/n): " c
echo ""

if [[ $c = n ]] || [[ $c = N ]]; then
    read -p "Define your tool's root directory here: " ROOT_DIR
fi

echo "Root directory is set to be: $ROOT_DIR."
echo "The rustGen command was added into PATH."
export ROOT_DIR
export PATH=${ROOT_DIR}:$PATH
export Strategy=ubfs
export GenChoose=unsafeHeuristic
echo "Environment initiation done!"
