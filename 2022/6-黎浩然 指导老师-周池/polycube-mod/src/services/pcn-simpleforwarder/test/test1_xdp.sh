#!/bin/bash

set -e
set -x

"${BASH_SOURCE%/*}/test1.sh" XDP_SKB
