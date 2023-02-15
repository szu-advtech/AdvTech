# fuzz-target-generator

 This project is our initial effort to automatically generator fuzz target based on API dependencies.

## Install

Currently supports rustc version: 1.58.0-nightly (91b931926 2021-10-23)
```
$ https://github.com/SZU-SE/Rust-Lib-Testing.git --recursive 
$ cd 
$ rustup component add rust-src rustc-dev llvm-tools-preview
$ export DOC_RUST_LANG_ORG_CHANNEL="https://doc.rust-lang.org/nightly"
$ cargo install --path .
```
