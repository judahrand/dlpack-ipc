use flatc_rust;

use std::path::Path;

fn main() {
    flatc_rust::run(flatc_rust::Args {
        inputs: &[Path::new("format/Tensor.fbs")],
        out_dir: Path::new("src/gen"),
        extra: &["--filename-suffix", ""],
        ..Default::default()
    }).expect("flatc");
}
