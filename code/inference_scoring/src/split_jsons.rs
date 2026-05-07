

use std::{collections::HashSet, fs, path::PathBuf};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{Map, Value};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short = 'a', long)]
    output_a: PathBuf,

    #[arg(short = 'b', long)]
    output_b: PathBuf,

    #[arg(short, long, value_name = "KEY")]
    duplicate: Vec<String>,

    #[arg(short, long, value_name = "KEY")]
    keep: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let raw = fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read {}", args.input.display()))?;
    let records: Vec<Value> = serde_json::from_str(&raw)
        .with_context(|| "Input must be a JSON array of objects")?;

    let dup_set: HashSet<&str> = args.duplicate.iter().map(String::as_str).collect();
    let keep_set: HashSet<&str> = args.keep.iter().map(String::as_str).collect();

    let mut out_a: Vec<Value> = Vec::with_capacity(records.len());
    let mut out_b: Vec<Value> = Vec::with_capacity(records.len());

    for rec in records {
        let obj = rec.as_object().cloned().context("Expected object items")?;

        let mut a_map = Map::new();
        let mut b_map = Map::new();

        for (k, v) in obj {
            let key_str = k.as_str();
            if dup_set.contains(key_str) {
                a_map.insert(k.clone(), v.clone());
                b_map.insert(k, v);
            } else if keep_set.contains(key_str) {
                a_map.insert(k, v);
            } else {
                b_map.insert(k, v);
            }
        }

        out_a.push(Value::Object(a_map));
        out_b.push(Value::Object(b_map));
    }

    fs::write(&args.output_a, serde_json::to_string_pretty(&out_a)?)
        .with_context(|| format!("Failed to write {}", args.output_a.display()))?;
    fs::write(&args.output_b, serde_json::to_string_pretty(&out_b)?)
        .with_context(|| format!("Failed to write {}", args.output_b.display()))?;

    println!(
        "Wrote {} records to {} and {}",
        out_a.len(),
        args.output_a.display(),
        args.output_b.display()
    );

    Ok(())
}
