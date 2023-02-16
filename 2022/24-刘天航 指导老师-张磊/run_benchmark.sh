#!/bin/bash

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DIRNAME=$1

node $BASEDIR/chrome/single_obj_test.js --log --dir $DIRNAME
node $BASEDIR/chrome/multi_obj_test.js --log --multi --dir $DIRNAME
