

use anyhow::{anyhow, Context, Result};
use chrono::Local;
use clap::Parser;
use governor::{Quota, RateLimiter};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map as JsonMap, Value};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::Duration as StdDuration;
use tokio::time::{sleep, Duration};
use tiktoken_rs::tiktoken::p50k_base;

#[derive(Parser, Debug)]
#[command(
    version,
    author,
    about = "Generate paraphrase variants with resume support, rate-limits and dynamic chunking"
)]
struct Cli {
    #[arg(long = "input", value_name = "INPUT_FILE")]
    input:  PathBuf,
    #[arg(long = "output", value_name = "OUTPUT_FILE")]
    output: PathBuf,

    #[arg(long, default_value = "all")]
    version_set: String,

    #[arg(long, default_value = "gemini-2.5-flash-preview-05-20")]
    model: String,

    #[arg(long, default_value_t = 5)]
    max_attempts: u8,

    #[arg(long = "api-call-maximum", default_value_t = 10_000)]
    api_call_maximum: usize,

    #[arg(long = "max-prompts-per-chunk", default_value_t = 11)]
    max_prompts_per_chunk: usize,

    #[arg(long = "log-name", default_value = "run")]
    log_name: String,

    #[arg(long = "api-key")]
    api_key: Option<String>,
}

static VERSION_SETS: phf::Map<&'static str, &'static [&'static str]> = phf::phf_map! {
    "all" => &[
        "instruct_output_markdown",
        "instruct_one_typo_punctuation",
        "instruct_coord_to_subord",
        "instruct_future_tense",
        "instruct_polite_request",
        "instruct_dramatic",
        "instruct_sardonic",
        "instruct_joke",
        "instruct_formal_memo",
        "instruct_double_negative",
        "instruct_leet_speak"
    ],
};

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Record {
    prompt_count:         u32,
    #[serde(alias = "instruction", alias = "instruction_original")]
    instruction_original: String,

    #[serde(flatten)]
    extra: JsonMap<String, Value>,
}

struct ModelLimits { input: usize  }
fn model_limits(name: &str) -> ModelLimits {
    match name {
        "gemini-2.5-flash-preview-05-20" => ModelLimits { input: 1_048_576 },
        "gemini-2.5-flash-lite-preview-06-17" => ModelLimits { input: 1_000_000 },
        "gemini-2.5-flash" => ModelLimits { input: 1_048_576 },
        "gemini-2.5-pro" => ModelLimits { input: 1_048_576 },
        _ => ModelLimits { input: 1_000_000 },
    }
}
fn rpm_for(name: &str) -> u32 {
    match name {
        "gemini-2.5-flash-preview-05-20"     => 10,
        "gemini-2.5-flash-lite-preview-06-17" => 15,
        _                                      => 5,
    }
}

struct Logger {
    writer: BufWriter<fs::File>,
}
impl Logger {
    fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = fs::OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { writer: BufWriter::new(file) })
    }
    fn log(&mut self, msg: &str) {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(self.writer, "[{ts}] {msg}");
        let _ = self.writer.flush();
    }
}

fn load_status<P: AsRef<Path>>(p: P) -> Result<HashMap<u32, bool>> {
    if p.as_ref().exists() {
        let f = fs::File::open(&p)?;
        Ok(serde_json::from_reader(f).context("cannot parse status file")?)
    } else {
        Ok(HashMap::new())
    }
}
fn save_status<P: AsRef<Path>>(p: P, map: &HashMap<u32, bool>) -> Result<()> {
    let f = fs::File::create(p)?;
    serde_json::to_writer_pretty(f, map)?;
    Ok(())
}

fn build_prompt(original: &str, keys: &[&str]) -> String {
    let bullet_list = keys
        .iter()
        .map(|k| format!("* **{k}** - rewrite in the \"{k}\" variant."))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "You are an expert paraphraser.\n\
         Rewrite the *Original Instruction* **once for every key** in the list below.\n\n\
         {bullet_list}\n\n\
         Each rewritten instruction must keep the *semantic request* identical to the original\n         (only style, tone, wording etc. may differ).\n\n\
         Return **ONE** JSON object with exactly those keys.\n\n\
         Original Instruction:\n{original}")
}

fn schema_for(keys: &[&str]) -> Value {
    let mut props = JsonMap::new();
    for k in keys { props.insert((*k).into(), json!({"type": "string"})); }
    json!({"type":"object", "properties": props, "required": keys})
}

async fn query_gemini(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    prompt: &str,
    schema: &Value,
) -> Result<JsonMap<String, Value>> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
    );
    let body = json!({
        "contents": [{"role":"user", "parts": [{"text": prompt }]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    });
    let resp = client.post(url).json(&body).send().await?;
    if !resp.status().is_success() {
        return Err(anyhow!("{} — {}", resp.status(), resp.text().await?));
    }
    let raw: Value = resp.json().await?;
    let text = raw["candidates"][0]["content"]["parts"][0]["text"].as_str()
        .ok_or_else(|| anyhow!("unexpected response structure: {}", raw))?;
    serde_json::from_str(text).map_err(|e| anyhow!("failed to parse model JSON: {e}\n{text}"))
}

fn read_input_records<P: AsRef<Path>>(p: P) -> Result<Vec<Record>> {
    let raw = fs::read_to_string(&p).with_context(|| format!("cannot read {}", p.as_ref().display()))?;
    Ok(serde_json::from_str(&raw).context("invalid JSON in input file")?)
}
fn write_output<P: AsRef<Path>>(p: P, all_records: &[Record]) -> Result<()> {
    let mut vec = all_records.to_vec();
    vec.sort_by_key(|r| r.prompt_count);
    let f = fs::File::create(p)?;
    serde_json::to_writer_pretty(f, &vec)?;
    Ok(())
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all("logs")?;

    let ts = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let logfile = PathBuf::from("logs").join(format!(
        "{}_{}_{}.log",
        cli.output.file_stem().unwrap().to_string_lossy(),
        cli.log_name,
        ts
    ));
    let mut log = Logger::new(&logfile)?;
    log.log(&format!("Run START. Model: {}", cli.model));

    let api_key = cli
        .api_key
        .or_else(|| std::env::var("GOOGLE_API_KEY").ok())
        .context("GOOGLE_API_KEY not set and --api-key not provided")?;

    let client = {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(180))
            .build()?
    };

    let rpm = rpm_for(&cli.model);
    let quota = Quota::with_period(StdDuration::from_secs(60))
        .expect("non-zero period")
        .allow_burst(NonZeroU32::new(rpm).unwrap());
    let limiter = RateLimiter::direct(quota);

    let status_path = PathBuf::from("logs").join(format!("id_status_{}.json", cli.log_name));
    let mut id_status = load_status(&status_path)?;

    let mut input_records = read_input_records(&cli.input)?;
    log.log(&format!("Loaded {} input records", input_records.len()));

    let mut output_map: HashMap<u32, Record> = if cli.output.exists() {
        let existing: Vec<Record> = serde_json::from_reader(fs::File::open(&cli.output)?)
            .context("failed to parse existing output")?;
        log.log(&format!("Loaded {} existing output records", existing.len()));
        existing.into_iter().map(|r| (r.prompt_count, r)).collect()
    } else { HashMap::new() };

    let bpe = p50k_base().expect("BPE load failed");
    let max_in_tokens = model_limits(&cli.model).input / 2; // we stay well under the hard cap

    let pb = ProgressBar::new(input_records.len() as u64);
    pb.set_style(ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - ID {msg}"
    )?);

    let mut api_calls_made = 0usize;
    let mut all_errors: HashMap<u32, Vec<String>> = HashMap::new();

    'record_loop: for rec in input_records.iter_mut() {
        let pid = rec.prompt_count;
        pb.set_message(format!("{}", pid));

        if let Some(done) = id_status.get(&pid) { if *done { pb.inc(1); continue; } }

        if let Some(prev) = output_map.get(&pid) {
            rec.extra.extend(prev.extra.clone());
        }

        let all_keys = VERSION_SETS
            .get(cli.version_set.as_str())
            .ok_or_else(|| anyhow!("Unknown version set {}", cli.version_set))?;
        let missing: Vec<_> = all_keys
            .iter()
            .copied()
            .filter(|k| !rec.extra.contains_key(*k))
            .collect();
        if missing.is_empty() {
            id_status.insert(pid, true);
            pb.inc(1);
            continue;
        }

        let mut batches = Vec::<Vec<&str>>::new();
        let mut current_batch = Vec::<&str>::new();
        let mut current_tokens = 0usize;
        for &k in &missing {
            let line = format!("* **{k}** - rewrite\n");
            let t = bpe.encode_with_special_tokens(&line).len();
            let would_overflow = current_tokens + t > max_in_tokens
                || current_batch.len() >= cli.max_prompts_per_chunk;
            if would_overflow && !current_batch.is_empty() {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_tokens = 0;
            }
            current_batch.push(k);
            current_tokens += t;
        }
        if !current_batch.is_empty() { batches.push(current_batch); }

        let mut success_any = false;
        for batch_keys in batches {
            if api_calls_made >= cli.api_call_maximum {
                log.log("[warn] API-call maximum reached, aborting run early");
                break 'record_loop;
            }

            let prompt = build_prompt(&rec.instruction_original, &batch_keys);
            let schema = schema_for(&batch_keys);

            api_calls_made += 1;
            limiter.until_ready().await; // RPM gate
            let mut success = false;
            for attempt in 1..=cli.max_attempts {
                match query_gemini(&client, &api_key, &cli.model, &prompt, &schema).await {
                    Ok(map) => {
                        for (k, v) in map { rec.extra.insert(k, v); }
                        success = true;
                        success_any = true;
                        break;
                    }
                    Err(e) => {
                        log.log(&format!(
                            "[error] ID {pid}: attempt {}/{} failed - {}",
                            attempt, cli.max_attempts, e
                        ));
                        if attempt < cli.max_attempts {
                            let secs = 3 * attempt as u64;
                            sleep(Duration::from_secs(secs)).await;
                        }
                    }
                }
            }
            if !success {
                let msg = format!("Batch {:?} failed after {} attempts", batch_keys, cli.max_attempts);
                all_errors.entry(pid).or_default().push(msg.clone());
                log.log(&format!("[fatal] ID {pid}: {msg}"));
            }
        }

        if success_any {
            output_map.insert(pid, rec.clone());
            let all_vec: Vec<_> = output_map.values().cloned().collect();
            write_output(&cli.output, &all_vec)?;
            id_status.insert(pid, true);
            save_status(&status_path, &id_status)?;
        }
        pb.inc(1);
    }

    pb.finish_with_message("All records processed");

    if all_errors.is_empty() {
        log.log("Run finished with **no fatal errors** - hooray!");
    } else {
        log.log(&format!("Run finished - {} prompt IDs had fatal errors:", all_errors.len()));
        for (pid, errs) in &all_errors {
            log.log(&format!("  ID {pid}:"));
            for e in errs { log.log(&format!("    - {e}")); }
        }
    }
    Ok(())
}
