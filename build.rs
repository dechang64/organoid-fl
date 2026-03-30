fn main() {
    tonic_build::configure()
        .compile_protos(&["proto/vectordb.proto"], &["proto"])
        .unwrap();
}
