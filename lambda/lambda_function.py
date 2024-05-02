import json
import boto3
import os
import rottnest
import boto3

def handler(event, context):
    print(event)
    print(os.listdir("/tmp"))

    embedding_name = os.getenv('EMBEDDING_NAME')

    if not os.path.exists(f"/tmp/{embedding_name}"):
        print("/tmp was cleared between invocs!")
        s3 = boto3.client('s3')
        s3.download_file('cluster-dump', embedding_name, f"/tmp/{embedding_name}")
    else:
        print("Found cached embeddings!")

    index_bucket = os.getenv('INDEX_BUCKET')
    
    try:
        query = event['queryStringParameters']['query']
    except KeyError:
        print('No query')
    
    try:
        K = int(event['queryStringParameters']['K'])
    except KeyError:
        print('No K')

    # let's say we search 10 index files
    result = rottnest.search_index_bm25([f"s3://{index_bucket}/merged_index"], query, K = K, query_expansion = "openai", cache_dir = "/tmp", reader_type = "aws")
    
    res = {
        "statusCode": 200,
        "headers": {
            "Content-Type": "*/*"
        },
        "body": "\n\n\n".join(result["text"].to_list())
    }

    return res