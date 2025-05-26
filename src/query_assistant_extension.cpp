#include "duckdb.h"
#include "duckdb/common/helper.hpp"
#include "duckdb/execution/expression_executor_state.hpp"
#include "duckdb/main/query_profiler.hpp"
#define DUCKDB_EXTENSION_MAIN

#include "query_assistant_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>
#include <duckdb/main/connection_manager.hpp>

#include "sql_code_generator.hpp"

namespace duckdb {

inline std::string get_ddl_statements(ExpressionState &state) {
	Connection conn(*(state.GetContext().db));
	auto query = conn.Query("SELECT sql from duckdb_tables();");
	std::string ddl_statement;
	for (idx_t i = 0; i < query->RowCount(); i++) {
		ddl_statement.append(query->GetValue(i, 0).ToString());
	}
	return ddl_statement;
}

inline void QueryAssistantScalarFunction(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &name_vector = args.data[0];
	UnaryExecutor::Execute<string_t, string_t>(name_vector, result, args.size(), [&](string_t name) {
		auto res = SqlCodeGeneratorSingleton::instance().generate(name.GetString(), get_ddl_statements(state));
		return StringVector::AddString(result, res);
	});
}

static unique_ptr<FunctionData> QueryAssistantBindingFunction(ClientContext &context, ScalarFunction &bound_function,
                                                              vector<unique_ptr<Expression>> &arguments) {
	if (arguments[0]->type != ExpressionType::VALUE_CONSTANT) {
		throw InvalidTypeException("input query is not a string!");
	}

	SqlCodeGeneratorSingleton::instance();
	return nullptr;
}

static void LoadInternal(DatabaseInstance &instance) {
	// Register a scalar function
	auto quack_scalar_function = ScalarFunction("query_assistant", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                                            QueryAssistantScalarFunction, QueryAssistantBindingFunction);
	ExtensionUtil::RegisterFunction(instance, quack_scalar_function);
}

void QueryAssistantExtension::Load(DuckDB &db) {
	LoadInternal(*db.instance);
}
std::string QueryAssistantExtension::Name() {
	return "query_assistant";
}

std::string QueryAssistantExtension::Version() const {
#ifdef EXT_VERSION_QUERY_ASSISTANT
	return EXT_VERSION_QUERY_ASSISTANT;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void query_assistant_init(duckdb::DatabaseInstance &db) {
	duckdb::DuckDB db_wrapper(db);
	db_wrapper.LoadExtension<duckdb::QueryAssistantExtension>();
}

DUCKDB_EXTENSION_API const char *query_assistant_version() {
	return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
