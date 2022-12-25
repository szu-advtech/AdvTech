#![feature(rustc_private)]
#![feature(box_patterns)]
#![allow(dead_code, unused_imports, unused_variables)]

extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustdoc;

use fuzz_target_generator::call_rust_doc_main;
use rustc_driver::{abort_on_err, describe_lints};
use rustc_errors::ErrorReported;
use std::process;
use std::time::Instant;

fn main() {
    let start = Instant::now();
    println!("My Rustdoc for Rust Libraries: v0.2.0");

    call_rust_doc_main();

    println!(
        "My Rustdoc exits successfully. Total time cost: {:?} ms",
        start.elapsed().as_millis()
    );
}
