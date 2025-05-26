// build.rs

fn main() {
    let _build = cxx_build::bridge("src/lib.rs");
}