#pragma once

#include <rust/cxx.h>
#include <sql_assistant_rust/lib.h>

class SqlCodeGeneratorSingleton {
public:
	SqlCodeGeneratorSingleton(const SqlCodeGeneratorSingleton &) = delete;
	SqlCodeGeneratorSingleton &operator=(const SqlCodeGeneratorSingleton &) = delete;

	static SqlCodeGeneratorSingleton &instance() {
		static SqlCodeGeneratorSingleton instance;
		return instance;
	}

	std::string generate(const std::string &prompt, const std::string &context) {
		return code_gen->generate(::rust::Str(prompt), ::rust::Str(context)).c_str();
	}

private:
	SqlCodeGeneratorSingleton() {
		code_gen = create_sql_code_generator().into_raw();
	}

private:
	SqlCodeGenerator *code_gen;
};
