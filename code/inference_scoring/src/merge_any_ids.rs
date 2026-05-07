

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
use clap::{Arg, Command};
use serde_json::Value;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("merge_random_ids")
        .version("1.0")
        .about("Merge multiple JSON array files into one, unique by `prompt_count`")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("Input JSON file (array). Can be used multiple times.")
                .required(true)
                .num_args(1..) // 1 or more
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output file path")
                .num_args(1)
        )
        .get_matches();

    let inputs: Vec<_> = matches
        .get_many::<String>("input")
        .unwrap()
        .map(|s| s.as_str())
        .collect();

    let mut all_objs: Vec<Value> = Vec::new();
    for fname in &inputs {
        let text = fs::read_to_string(fname)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", fname, e));
        let part: Vec<Value> = serde_json::from_str(&text)
            .unwrap_or_else(|_| panic!("{} is not a JSON array", fname));
        all_objs.extend(part);
    }
    eprintln!("Collected {} objects from {} file(s)", all_objs.len(), inputs.len());

    let mut by_id: BTreeMap<u64, Value> = BTreeMap::new();
    for obj in all_objs {
        if let Value::Object(ref map) = obj {
            if let Some(Value::Number(n)) = map.get("prompt_count") {
                if let Some(id) = n.as_u64() {
                    by_id.entry(id).or_insert(obj);
                    continue;
                }
            }
        }
    }

    let unique_sorted: Vec<Value> = by_id.into_iter().map(|(_, v)| v).collect();
    eprintln!("Reduced to {} unique prompt_count IDs", unique_sorted.len());

    let out_path = if let Some(o) = matches.get_one::<String>("output") {
        PathBuf::from(o)
    } else {
        let first = Path::new(&inputs[0]);
        let parent = first.parent().unwrap_or_else(|| Path::new("."));
        let stem = first
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("merged");
        let ext = first
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("json");
        let base = stem.split_once('_').map(|(a, _)| a).unwrap_or(stem);
        parent.join(format!("{base}_merged.{ext}"))
    };

    fs::write(&out_path, serde_json::to_string_pretty(&unique_sorted)?)?;
    println!("Wrote merged file → {}", out_path.display());

    Ok(())
}
