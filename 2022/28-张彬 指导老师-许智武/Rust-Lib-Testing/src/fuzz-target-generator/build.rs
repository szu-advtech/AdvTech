// build.rs

fn main() {
    println!("cargo:rustc-env=DOC_RUST_LANG_ORG_CHANNEL=https://doc.rust-lang.org/nightly");
}