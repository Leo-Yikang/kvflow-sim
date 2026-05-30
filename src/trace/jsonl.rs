use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{KvFlowError, Result};

use super::LlmRequest;

pub fn read_jsonl(path: impl AsRef<Path>) -> Result<Vec<LlmRequest>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut requests = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line_no = idx + 1;
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: LlmRequest =
            serde_json::from_str(trimmed).map_err(|err| KvFlowError::InvalidTraceLine {
                line: line_no,
                message: err.to_string(),
            })?;
        requests.push(req);
    }

    Ok(requests)
}

pub fn write_jsonl(path: impl AsRef<Path>, requests: &[LlmRequest]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for req in requests {
        serde_json::to_writer(&mut writer, req)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_and_write_jsonl_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let requests = vec![LlmRequest {
            request_id: 1,
            session_id: 7,
            turn_id: 0,
            arrival_ns: 10,
            prompt_tokens: 1024,
            new_prompt_tokens: 1024,
            output_tokens: 128,
            model_id: "llama-8b".to_string(),
            slo_ttft_ns: None,
            slo_tbt_ns: Some(50_000_000),
        }];

        write_jsonl(&path, &requests).unwrap();
        let loaded = read_jsonl(&path).unwrap();

        assert_eq!(loaded, requests);
    }
}
