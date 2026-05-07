
use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
};

use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short, long)]
    count: usize,

    #[arg(short = 's', long = "start-id")]
    start_id: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct Item {
    instruction_original: String,
    subject: String,
    choices: Vec<String>,
    answer: serde_json::Value, // could be usize or String in other files
    prompt_count: usize,
    split: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut raw = String::new();
    File::open(&args.input)?.read_to_string(&mut raw)?;
    let mut items: Vec<Item> = serde_json::from_str(&raw)?;

    let keep = args.count.min(items.len());
    let mut selected: Vec<Item> = items.drain(0..keep).collect();
    for (i, item) in selected.iter_mut().enumerate() {
        item.prompt_count = args.start_id + i;
    }

    let mut combined: Vec<Item> = if args.output.exists() {
        let mut existing_raw = String::new();
        File::open(&args.output)?.read_to_string(&mut existing_raw)?;

        if existing_raw.trim().is_empty() {
            Vec::new() // empty file → nothing to merge
        } else {
            serde_json::from_str(&existing_raw)?
        }
    } else {
        Vec::new()
    };

    combined.extend(selected);

    let pretty = serde_json::to_string_pretty(&combined)?;
    File::create(&args.output)?.write_all(pretty.as_bytes())?;

    println!(
        "Appended {keep} item(s) to {}, prompt_count starting at {}. \
         File now holds {} item(s).",
        args.output.display(),
        args.start_id,
        combined.len()
    );
    Ok(())
}
