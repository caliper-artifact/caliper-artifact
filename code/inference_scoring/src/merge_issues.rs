

use std::{env, fs};
use serde_json::{Value, json};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);

    let out_file = args.next().ok_or("Missing <OUT_FILE> argument")?;

    if args.len() == 0 {
        eprintln!("Usage: merge_json <OUT_FILE> <IN_FILE_1> <IN_FILE_2> [...]");
        std::process::exit(1);
    }

    let mut merged: Vec<Value> = Vec::new();

    for path in args {
        let data = fs::read_to_string(&path)
            .map_err(|e| format!("Cannot read '{}': {}", path, e))?;
        let json: Value = serde_json::from_str(&data)
            .map_err(|e| format!("'{}' is not valid JSON: {}", path, e))?;

        match json {
            Value::Array(items) => merged.extend(items),
            other => {
                return Err(format!("'{}' must contain a top‑level JSON array, found {}", path, other).into());
            }
        }
    }

    let serialized = serde_json::to_string_pretty(&json!(merged))?;
    fs::write(&out_file, serialized)?;

    println!("✔ Merged {} items → {}", merged.len(), out_file);
    Ok(())
}
