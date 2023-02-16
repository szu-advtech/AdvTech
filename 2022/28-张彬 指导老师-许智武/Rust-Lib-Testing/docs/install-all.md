## Install

### Install the tool dependencies

#### Ubuntu

```
sudo apt install python3 expect
```

#### MacOS

```
brew install python3 expect
```

### Download the Tool and Rust Compiler Source Code

Clone the repository to a local directory. 
Note that the compilation depends on external submodules, you also need to synchronize submodules.
Try following commands:
```
$ git clone https://github.com/SZU-SE/Rust-Lib-Testing.git
$ cd Rust-Lib-Testing
$ git submodule init
$ git submodule update
```

### One-click build

For simplicity, we provide shell script for the whole installation.
```
$ scripts/build.sh
```

### Setup step-by-step

#### Switch to Nightly Toolchain

Switch the rust compiler to lastest nightly toolchain
```
$ rustup install nightly
$ rustup default nightly
$ rustup update
$ rustup component add rust-src rustc-dev llvm-tools-preview
```

#### Pollute the librustdoc

The following script will make librustdoc can be use outside.
```
scripts/rs_pollution.py -f src/rust/src/librustdoc
```

#### rust-operation-sequence-analyzer

This component can only work with an older versions of rust complier currently. So you need to run `rustup component add rust-src rustc-dev llvm-tools-preview` again under its working folder.
```
$ cd src/rust-operation-sequence-analyzer
$ rustup component add rust-src rustc-dev llvm-tools-preview
$ cargo install --path .
```

Move to test examples, and run with cargo subcommands
```
$ cd src/rust-operation-sequence-analyzer/examples/queue
$ cargo clean
$ cargo rosa
```

You need to run
```
cargo clean
```
before re-detecting.

#### fuzz-target-generator

```
$ cd src/fuzz-target-generator
$ DOC_RUST_LANG_ORG_CHANNEL="https://doc.rust-lang.org/nightly" cargo install --path .
```

### Usage

You need to import environment variable LD_LIBRARY_PATH before using the tool.

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(rustc --print sysroot)/lib
```

The src/fuzz-target-generator/target/release/fuzz-target-generator is currently the same as rustdoc.

```
src/fuzz-target-generator/target/release/fuzz-target-generator --help
```

To be continued.