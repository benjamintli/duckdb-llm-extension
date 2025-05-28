use core::fmt;

use anyhow::Error;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::qwen2::{Config, ModelForCausalLM},
};
use chat_templates::{apply_template, Message};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

const SYSTEM_PROMPT: &str = r#"System:
Your task is to generate valid DuckDB SQL to answer the question that the user asks. You should only respond with a valid DuckDB SQL query.

Here are some DuckDB SQL syntax specifics you should be aware of:


- DuckDB use double quotes (") for identifiers that contain spaces or special characters, or to force case-sensitivity and single quotes (') to define string literals
- DuckDB can query CSV, Parquet, and JSON directly without loading them first, e.g. `SELECT * FROM 'data.csv';`
- DuckDB supports CREATE TABLE AS (CTAS): `CREATE TABLE new_table AS SELECT * FROM old_table;`
- DuckDB queries can start with FROM, and optionally omit SELECT *, e.g. `FROM my_table WHERE condition;` is equivalent to `SELECT * FROM my_table WHERE condition;`
- DuckDB allows you to use SELECT without a FROM clause to generate a single row of results or to work with expressions directly, e.g. `SELECT 1 + 1 AS result;`
- DuckDB supports attaching multiple databases, unsing the ATTACH statement: `ATTACH 'my_database.duckdb' AS mydb;`. Tables within attached databases can be accessed using the dot notation (.), e.g. `SELECT * FROM mydb.table_name syntax`. The default databases doesn't require the do notation to access tables. The default database can be changed with the USE statement, e.g. `USE my_db;`.
- DuckDB is generally more lenient with implicit type conversions (e.g. `SELECT '42' + 1;` - Implicit cast, result is 43), but you can always be explicit using `::`, e.g. `SELECT '42'::INTEGER + 1;`
- DuckDB can extract parts of strings and lists using [start:end] or [start:end:step] syntax. Indexes start at 1. String slicing: `SELECT 'DuckDB'[1:4];`. Array/List slicing: `SELECT [1, 2, 3, 4][1:3];`
- DuckDB has a powerful way to select or transform multiple columns using patterns or functions. You can select columns matching a pattern: `SELECT COLUMNS('sales_.*') FROM sales_data;` or transform multiple columns with a function: `SELECT AVG(COLUMNS('sales_.*')) FROM sales_data;`
- DuckDB an easy way to include/exclude or modify columns when selecting all: e.g. Exclude: `SELECT * EXCLUDE (sensitive_data) FROM users;` Replace: `SELECT * REPLACE (UPPER(name) AS name) FROM users;`
- DuckDB has a shorthand for grouping/ordering by all non-aggregated/all columns. e.g `SELECT category, SUM(sales) FROM sales_data GROUP BY ALL;` and `SELECT * FROM my_table ORDER BY ALL;`
- DuckDB can combine tables by matching column names, not just their positions using UNION BY NAME. E.g. `SELECT * FROM table1 UNION BY NAME SELECT * FROM table2;`
- DuckDB has an inutitive syntax to create List/Struct/Map and Array types. Create complex types using intuitive syntax. List: `SELECT [1, 2, 3] AS my_list;`, Struct: `{{'a': 1, 'b': 'text'}} AS my_struct;`, Map: `MAP([1,2],['one','two']) as my_map;`. All types can also be nested into each other. Array types are fixed size, while list types have variable size. Compared to Structs, MAPs do not need to have the same keys present for each row, but keys can only be of type Integer or Varchar. Example: `CREATE TABLE example (my_list INTEGER[], my_struct STRUCT(a INTEGER, b TEXT), my_map MAP(INTEGER, VARCHAR),  my_array INTEGER[3], my_nested_struct STRUCT(a INTEGER, b Integer[3]));`
- DuckDB has an inutive syntax to access struct fields using dot notation (.) or brackets ([]) with the field name. Maps fields can be accessed by brackets ([]).
- DuckDB's way of converting between text and timestamps, and extract date parts. Current date as 'YYYY-MM-DD': `SELECT strftime(NOW(), '%Y-%m-%d');` String to timestamp: `SELECT strptime('2023-07-23', '%Y-%m-%d')::TIMESTAMP;`, Extract Year from date: `SELECT EXTRACT(YEAR FROM DATE '2023-07-23');`
- Column Aliases in WHERE/GROUP BY/HAVING: You can use column aliases defined in the SELECT clause within the WHERE, GROUP BY, and HAVING clauses. E.g.: `SELECT a + b AS total FROM my_table WHERE total > 10 GROUP BY total HAVING total < 20;`
- DuckDB allows generating lists using expressions similar to Python list comprehensions. E.g. `SELECT [x*2 FOR x IN [1, 2, 3]];` Returns [2, 4, 6].
- DuckDB allows chaining multiple function calls together using the dot (.) operator. E.g.: `SELECT 'DuckDB'.replace('Duck', 'Goose').upper(); -- Returns 'GOOSEDB';`
- DuckDB has a JSON data type. It supports selecting fields from the JSON with a JSON-Path expression using the arrow operator, -> (returns JSON) or ->> (returns text) with JSONPath expressions. For example: `SELECT data->'$.user.id' AS user_id, data->>'$.event_type' AS event_type FROM events;`
- DuckDB has built-in functions for regex regexp_matches(column, regex), regexp_replace(column, regex), and regexp_extract(column, regex).
- DuckDB has a way to quickly get a subset of your data with `SELECT * FROM large_table USING SAMPLE 10%;`
"#;

pub struct SqlCodeGenerator {
    model: ModelForCausalLM,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

fn get_device() -> Result<Device, Error> {
    #[cfg(target_os = "macos")]
    return Ok(Device::new_metal(0)?);
    #[cfg(not(target_os = "macos"))]
    return Ok(Device::Cpu);
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
        let device = get_device()?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
        let model = ModelForCausalLM::new(&config, vb)?;
        let logits_processor = LogitsProcessor::new(299792458, Some(0.0), None);
        Ok(Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty: 1.10,
            repeat_last_n: 64,
            device: device,
        })
    }

    pub fn generate(&mut self, prompt: &str, table_schema: &str) -> Result<String, Error> {
        self.tokenizer.clear();
        let combined_prompt = format!("{}\nSCHEMA: {}", prompt, table_schema);
        let system_message = Message {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        };
        let user_message = Message {
            role: "user".to_string(),
            content: combined_prompt,
        };
        let chat_template = apply_template(
            chat_templates::ChatTemplate::ChatML,
            &vec![system_message, user_message],
            true,
        )?;
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
        self.model.clear_kv_cache();
        Ok(output)
    }
}
