#!/usr/bin/env bash

# this script's location
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


if [ -z "$1" ]; then
        echo "No detecting directory is provided"
        exit 1
fi
cargo build --release
export RUSTC=${PWD}/target/release/rust-operation-sequence-analyzer
export RUST_BACKTRACE=full
#export RUST_DETECTOR_TYPE=DoubleLockDetector
#export RUST_DETECTOR_TYPE=ConflictLockDetector
export RUST_DETECTOR_TYPE=OperationSequenceDetector
export RUST_DETECTOR_BLACK_LISTS="cc"
#export RUST_DETECTOR_WHITE_LISTS="inter,intra,static_ref"

cargo_dir_file=$(realpath $DIR/cargo_dir.txt)
rm $cargo_dir_file
touch $cargo_dir_file

pushd "$1" > /dev/null
cargo clean
cargo_tomls=$(find . -name "Cargo.toml")
for cargo_toml in ${cargo_tomls[@]}
do
#	echo $cargo_toml
        echo $(dirname $cargo_toml) >> $cargo_dir_file
done

IFS=$'\n' read -d '' -r -a lines < ${cargo_dir_file}
for cargo_dir in ${lines[@]}
do
        echo ${cargo_dir}
        pushd ${cargo_dir} > /dev/null
        cargo build
        popd > /dev/null
done
popd > /dev/null

#pushd "$1" > /dev/null
#cargo clean
#cargo check
#popd > /dev/null
