# Rottnest : Data Lake Indices

You have your data already in Parquet format in Iceberg or Delta (or json.gz, CSV ,etc). Unfortunately you need to do some stuff like full text search or vector search, so you have to ETL into Clickhouse, ElasticSearch or some vector database. It is complicated and expensive. Now you have an alternative.

## Installation

Local installation: `pip install rottnest`

Kubernetes Operator (upcoming)

## How to use

Build indices on your Parquet files, merge them, and query them. Very simple. Let's walk through a very simple example.

### BM25

`rottnest.index_file_bm25(f"msmarco/chunk_{i}.parquet","body", name = f"msmarco_index/{i}")`

`rottnest.merge_index_bm25("msmarco_index/merged", [f"msmarco_index/{i}" for i in range(1,3)])`

`result = rottnest.search_index_bm25(["msmarco_index/merged"], "politics", K = 10,query_expansion = "openai")`

### Regex

### Vector

![Architecture](assets/arch.png)

## Development

### Build Python wheel
```bash
maturin develop --features py
```
