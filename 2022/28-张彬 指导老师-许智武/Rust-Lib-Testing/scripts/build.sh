#!/usr/bin/env bash

# For Mac
if [ $(command uname) == "Darwin" ]; then
	if ! [ -x "$(command -v greadlink)"  ]; then
		brew install coreutils
	fi
	BIN_PATH=$(greadlink -f "$0")
	ROOT_DIR=$(dirname $(dirname $BIN_PATH))
# For Linux
else
	BIN_PATH=$(readlink -f "$0")
	ROOT_DIR=$(dirname $(dirname $BIN_PATH))
fi

export ROOT_DIR=${ROOT_DIR}
export DOC_RUST_LANG_ORG_CHANNEL="https://doc.rust-lang.org/nightly"
cd ${ROOT_DIR}

MODE="init"

if [ $# -ne 0 ];then
	if [ "$1" == "fast" ];then
		MODE="fast"
	elif [ "$1" == "local" ];then
		MODE="local"
	else
		MODE="init"
	fi
fi

set -euxo pipefail

# git submodule
if [ "$MODE" == "init" ];then
	cp ${ROOT_DIR}/src/rust/.gitmodules .gitmodules.copy
	cd ${ROOT_DIR}/src/rust
	git restore .
	python3 ${ROOT_DIR}/scripts/modifyGitUrl.py
	git submodule init
	git submodule update
fi

cd ${ROOT_DIR}

# Find python3 in your OS
PY_VERSION_FIRST_ATTEMPT=`python3 -V 2>&1|awk '{print $2}'|awk -F '.' '{print $1}'`
PY_VERSION_SECOND_ATTEMPT=`python -V 2>&1|awk '{print $2}'|awk -F '.' '{print $1}'`
if (( $PY_VERSION_FIRST_ATTEMPT == 3 ));then
	echo "Your 'python3' version is 3."
	PYTHONBIN="python3"
elif (( $PY_VERSION_SECOND_ATTEMPT == 3 ));then
	echo "Your 'python' version is 3."
	PYTHONBIN="python"
else
	echo "I don't know your 'python' Version. Try to use /usr/bin/python3"
	PYTHONBIN=""
fi

# auto config
if [ "$MODE" == "init" ] || [ "$MODE" == "local" ];then
	${PYTHONBIN} scripts/rs_pollution.py -f src/rust/src/librustdoc
fi
if [ "$MODE" == "init" ] || [ "$MODE" == "fast" ];then
	${PYTHONBIN} scripts/autoConfig.py
fi

if [ "$MODE" == "init" ];then
	cd ${ROOT_DIR}
	cargo install --force afl
	cargo install --force cargo-fuzz
fi


# Building incomplete prototypes with local compilers (test)
if [ "$MODE" == "local" ];then
	# Switch to Nightly Toolchain (lastest nightly may can not use to compile our tool, just use it for local debuging)
	# rustup install nightly
	# rustup default nightly
	# rustup update
	# rustup component add rust-src rustc-dev llvm-tools-preview

	# Build rust-operation-sequence-analyzer
	cd ${ROOT_DIR}/src/rust-operation-sequence-analyzer
	rustup component add rust-src rustc-dev llvm-tools-preview
	cargo install  --force --path .

	# Build fuzz-target-generator
	cd ${ROOT_DIR}/src/fuzz-target-generator
	rustup component add rust-src rustc-dev llvm-tools-preview
	cargo install --force --path .

	echo "Finish local test."
	exit 0

# Install the compiler
elif [ "$MODE" == "init" ] || [ "$MODE" == "fast" ];then
	
	cd ${ROOT_DIR}/src/rust
	./x.py build --stage 2
	
	# If you have tried "local" mode, the rustup toolchain must be set
	rustup toolchain link myrust ${ROOT_DIR}/src/rust/build/x86*/stage2
	rustup default myrust

	# install fuzz-deployer
	if [ "$MODE" == "init" ];then
		cd ${ROOT_DIR}/src/fuzz-deployer
		cargo install --path fuzzer_scripts
		cargo install --path find_literal
	fi

	# install fuzz-deployer, cargo-fuzz and afl.rs
	echo "Finish installation."
	exit 0
fi
