use sql_code_generator::SqlCodeGenerator;

mod sql_code_generator;

pub fn create_sql_code_generator() -> Box<SqlCodeGenerator> {
    Box::new(SqlCodeGenerator::new().expect("Failed to construct"))
}

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        type SqlCodeGenerator;
        fn create_sql_code_generator() -> Box<SqlCodeGenerator>;
        fn generate(self: &mut SqlCodeGenerator, prompt: &str, table_schemas:&str) -> Result<String>;
    }
}
