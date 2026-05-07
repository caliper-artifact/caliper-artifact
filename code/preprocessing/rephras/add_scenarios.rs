

use std::io::{self, Read};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut raw = String::new();
    io::stdin().read_to_string(&mut raw)?;

    let mut data: serde_json::Value = serde_json::from_str(&raw)?;

    let arr = data
        .as_array_mut()
        .ok_or("Input JSON must be an array of objects")?;

    for obj in arr {
        let map = obj
            .as_object_mut()
            .ok_or("Array element is not a JSON object")?;

        if let Some(instruction) = map
            .get("instruction_original")
            .and_then(|v| v.as_str())
        {
            if let Some(start) = instruction.find("Scenario 1") {
                let scenarios = &instruction[start..];
                map.insert(
                    "scenarios".into(),
                    serde_json::Value::String(scenarios.to_string()),
                );
            }
        }
    }

    println!("{}", serde_json::to_string_pretty(&data)?);
    Ok(())
}
