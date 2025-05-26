use core::fmt;

use anyhow::Error;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::qwen2::{Config, ModelForCausalLM},
};
use chat_templates::{apply_template, Message};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct SqlCodeGenerator {
    model: ModelForCausalLM,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl SqlCodeGenerator {
    pub fn new() -> Result<Self, Error> {
        let api = Api::new()?;
        let model_id = "benjamintli/duckdb-sqlcoder-0.5B".to_string();
        let repo = api.repo(Repo::with_revision(
            model_id,
            RepoType::Model,
            "main".to_string(),
        ));
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let filenames = vec![repo.get("model.safetensors")?];
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;

        let config_file = repo.get("config.json")?;
        let dtype = DType::F32;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &Device::Cpu)? };
        let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let model = ModelForCausalLM::new(&config, vb)?;
        let logits_processor = LogitsProcessor::new(299792458, Some(0.0), None);
        Ok(Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty: 1.10,
            repeat_last_n: 64,
            device: Device::Cpu,
        })
    }

    pub fn generate(&mut self, prompt: &str, table_schema: &str) -> Result<String, Error> {
        self.tokenizer.clear();
        let combined_prompt = format!("Write an SQL query based on the user's request. Pay attention to casing.\n{}\nSCHEMA: {}", prompt, table_schema);
        let message = Message {
            role: "user".to_string(),
            content: combined_prompt,
        };
        let chat_template =
            apply_template(chat_templates::ChatTemplate::ChatML, &vec![message], true)?;
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(chat_template, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
        let mut output = String::new();

        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let eos_token2 = match self.tokenizer.get_token("<|im_end|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|im_end|> token"),
        };
        for index in 0..256 {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            if next_token == eos_token || next_token == eos_token2 {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                output.push_str(&t);
            }
        }
        if let Some(rest) = self.tokenizer.decode_rest().map_err(Error::msg)? {
            output.push_str(&rest);
        }
        Ok(output)
    }
}
