

use std::{collections::{HashMap, HashSet}, fs, path::PathBuf};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{Map, Value};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long, required = true)]
    shared: Vec<String>,

    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut file_maps: Vec<HashMap<i64, Map<String, Value>>> = Vec::new();

    for path in &args.inputs {
        let data = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let array: Vec<Value> = serde_json::from_str(&data)
            .with_context(|| format!("{} is not valid JSON array", path.display()))?;

        let mut map = HashMap::new();
        for obj in array {
            let mut obj_map = obj.as_object().cloned().context("Expected JSON objects")?;
            if !obj_map.contains_key("style") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    obj_map.insert("style".into(), Value::String(stem.to_string()));
                }
            }
            let prompt_count = obj_map.get("prompt_count")
                .and_then(Value::as_i64)
                .context("`prompt_count` must be an integer")?;

            if map.insert(prompt_count, obj_map).is_some() {
                eprintln!("Warning: duplicate prompt_count {} in {} — keeping last", prompt_count, path.display());
            }
        }
        file_maps.push(map);
    }

    let mut common: Option<HashSet<i64>> = None;
    for fmap in &file_maps {
        let keys: HashSet<i64> = fmap.keys().copied().collect();
        common = match common {
            None => Some(keys),
            Some(acc) => Some(&acc & &keys),
        };
    }
    let common = common.unwrap_or_default();

    for (idx, fmap) in file_maps.iter().enumerate() {
        for key in fmap.keys() {
            if !common.contains(key) {
                eprintln!("File {} lacks prompt_count {} present in others; skipping.", args.inputs[idx].display(), key);
            }
        }
    }

    let mut merged: Vec<Value> = Vec::new();
    for &pc in &common {
        let mut combined = Map::new();
        combined.insert("prompt_count".into(), Value::from(pc));

        for fmap in &file_maps {
            if let Some(obj) = fmap.get(&pc) {
                for (k, v) in obj {
                    if args.shared.contains(k) {
                        if let Some(prev) = combined.get(k) {
                            if prev != v {
                                eprintln!("Conflict on shared key '{}' for prompt_count {}: choosing first value", k, pc);
                            }
                        } else {
                            combined.insert(k.clone(), v.clone());
                        }
                    } else {
                        combined.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        merged.push(Value::Object(combined));
    }

    merged.sort_by_key(|v| v.get("prompt_count").and_then(Value::as_i64).unwrap_or(i64::MAX));

    let json_out = serde_json::to_string_pretty(&merged)?;
    fs::write(&args.output, json_out)
        .with_context(|| format!("Failed to write {}", args.output.display()))?;

    println!("Merged {} prompt groups into {}", merged.len(), args.output.display());
    Ok(())
}
