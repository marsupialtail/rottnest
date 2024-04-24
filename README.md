# Rottnest : Data Lake Indices

You don't need ElasticSearch or some vector database to do full text search or vector search. Parquet + Rottnest is all you need. Rottnest is like Postgres indices for Parquet. 

## Installation

Local installation: `pip install rottnest`

Kubernetes Operator (upcoming)

## How to use

Build indices on your Parquet files, merge them, and query them. Very simple. Let's walk through a very simple example, in `demo.py`. It builds a BM25 index on two Parquet files, merges the indices, and searches the merged index for records related to cell phones. The code is here:

```
import rottnest
rottnest.index_file_bm25("example_data/0.parquet", "body", "index0")
rottnest.index_file_bm25("example_data/1.parquet", "body", "index1")
rottnest.merge_index_bm25("merged_index", ["index0", "index1"])
result = rottnest.search_index_bm25(["merged_index"], "cell phones", K = 10)
```

This code will still work if the Parquet files are in fact **on object storage**. You can copy the data files to an S3 bucket, say `s3://example_data/`. Then the following code will work:

```
import rottnest
rottnest.index_file_bm25("s3://example_data/0.parquet", "body", "index0")
rottnest.index_file_bm25("s3://example_data/1.parquet", "body", "index1")
rottnest.merge_index_bm25("merged_index", ["index0", "index1"])
result = rottnest.search_index_bm25(["merged_index"], "cell phones", K = 10)
```

It will use the index to search against the Parquet files on S3 directly. Rottnest has its own Parquet reader that makes this very very efficient.

Rottnest not only supports BM25 indices but also other indices, like regex and vector searches. More documentation will be forthcoming.

### Regex

### Vector

## Architecture

![Architecture](assets/arch.png)

## Development

### Build Python wheel
```bash
maturin develop --features "py,opendal"
```
or 
```bash
maturin develop --features "py,aws_sdk"
```
