#[cfg(feature = "logcloud")]
fn main() {
    use std::env;
    use std::path::PathBuf;

    let dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = PathBuf::from(dir).join("src").join("lava").join("logcloud");

    // Specify the directory containing the .a files
    println!("cargo:rustc-link-search=native={}", path.display());

    // Link against Compressor.a
    println!("cargo:rustc-link-lib=static=Compressor");

    // Link against Trainer.a
    println!("cargo:rustc-link-lib=static=Trainer");

    // Link against C++ standard library
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // Rerun the build script if the static libraries change
    println!("cargo:rerun-if-changed=src/lava/logcloud/libCompressor.a");
    println!("cargo:rerun-if-changed=src/lava/logcloud/libTrainer.a");
}

#[cfg(not(feature = "logcloud"))]
fn main() {}
