# DuckDB Text2SQL Extension

## Description
* This is a text2sql extension for DuckDB databases
* The intention is to be able to call a scalar function in DuckDB with a natural language query, and have it generate SQL!
* It uses the table info from an internal call to duckdb_tables() to fetch the DDL data on the table structure and feed it to the LLM 


## Usage
```
LOAD <path to extension from github actions>

SELECT query_assistant('find me all the customers who live in Canada from ontime table who have subscription dates later than 2021');
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ query_assistant('find me all the customers who live in Canada from ontime table who have subscription dates later than 2021') │
│                                                            varchar                                                            │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ SELECT * FROM ontime WHERE Country = 'Canada' AND Subscription_Date > '2021-01-01';
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```