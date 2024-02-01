# Rottnest : Machine Data -> LLMs
Rottnest aims to be the missing link between machine data (metrics, logs, traces) and large language models. The goal is to **empower** users to build retrieval-augmented generation (RAG) applications based off observability and cybersecurity data, such as root-cause debugging agent or AI SOC agent. While it's primarily focused on machine data, users might find it useful for other applications, like data curation for training LLMs.

**It is currently a work in progress, expected release date Q2/Q3 2024** Slowly migrating from https://github.com/marsupialtail/rottnest-cpp.

## The Indexing Story

**Logs** (Q2 2024): Unlike solutions like ElasticSearch which aims to lock all of your data in a format only ElasticSearch can read, Rottnest **indexes** your logs where they currently live, in gzipped JSON (AWS Cloudtrail / Datadog archives), Parquet (AWS Security Lake or Hive), Apache Iceberg and Delta Lake. 

- The indices are typically **100x smaller** than the raw data itself, and are meant to reside **100% on object storage**. This minimizes storage cost, while ElasticSearch indices can be as big as the source data itself.
- The indices support highly efficient **keyword-search, full text search, JSON search** (Snowflake VARIANT) and **vector search** (because why not), all from your laptop.
- Build indices on demand or as a long running job that track a data source.

**CLI**: `rottnest-index --format json.gz --source s3://cloudtrail_bucket/ --indices '{all: json}' --interval 30 --index_location s3://rottnest-index/`
**Kube**: `kubectl deploy -f rottnest-index.yaml`

**Metrics** (Q3 2024): We aim to expose a PromQL remote write endpoint to ingest metrics and convert them into Parquet format. Unlike logs, we do choose to store a copy of your metrics. We believe this to be a better approach than trying to read blocks from Thanos, e.g. since you might want to do this anyway to work with other SQL-speaking systems. On metrics, we aim to support indices that support:

- **Efficient metrics retrieval** on a Parquet-based storage format to support efficient SQL analytics on large volumes of metrics.
- **Fast outlier search**, **anomaly search** and **nearest-neighbor search**.

## The Retrieval Story

With an appropriate index, Rottnest aims to support efficient search with both **Snowflake-style SQL** and **Elastic-compatible API**. It is designed to support interactive search from your laptop on terabytes of indexed data. Under the hood, the searcher uses the index to avoid full table scans, avoiding the exorbitant costs of solutions like AWS Athena / Cloudtrail Lake.

**CLI**: `rottnest-search --source s3://cloudtrail_bucket --index_location s3://rottnest-index/ --sql "select * from table where responseElements:ConsoleLogin = 'Failure'"`

However Rottnest aims to support more retrieval APIs to **support LLM use-cases**. FOr example, Rottnest aims to have native retrieval APIs to detect:

- **Anomaly Detection**: retrieve windows of anomalous log messages or metrics
- **Pattern Matching**: retrieve sequences of log events like Splunk transactions or SQL Match Recognize, or detect patterns in metrics like spikes or dips
- **Vector Search**: retrieve windows of logs based on embedding search (Rottnest will automatically embed logs based on a simple rule-based embedding model, but more complicated models could be used)
- **Metrics to text**: come up with a natural language description for a time window of metrics to feed into LLM, e.g. service XYZ saw a spike in CPU utilization around 19:00 from the metrics data.
