

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::Value;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    #[arg(short, long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut merged_items = Vec::<Value>::new();

    for path in &args.inputs {
        let content = fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let json: Value = serde_json::from_str(&content)
            .with_context(|| format!("parsing {}", path.display()))?;

        match json {
            Value::Array(arr) => merged_items.extend(arr),
            other => bail!(
                "File {} is not a JSON array (found {:?})",
                path.display(),
                other
            ),
        }
    }

    let pretty = serde_json::to_string_pretty(&Value::Array(merged_items))?;
    fs::write(&args.output, pretty)
        .with_context(|| format!("writing {}", args.output.display()))?;

    println!(
        "Merged {} file(s) into {}",
        args.inputs.len(),
        args.output.display()
    );
    Ok(())
}
