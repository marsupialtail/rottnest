import rottnest
import pyarrow
from tqdm import tqdm
import polars

metadata = polars.read_parquet("bench.parquet")[:10]

# metadatas = []
# filenames = []
# for i in tqdm(range(10)):
#     filename = f"s3://redpajama-1t/c4/c4-train.0000{i}-of-01024.parquet"
#     x, y = rottnest.rottnest.get_parquet_layout("text", filename, "aws")
#     filenames.append(filename)
#     metadatas.append(y.metadata_bytes)

# polars.from_dict({"filename": filenames, "metadata_bytes": metadatas}).write_parquet("metadata.parquet")

file_metadata = polars.read_parquet("metadata.parquet")
file_metadata = {filename: metadata for filename, metadata in zip(file_metadata["filename"], file_metadata["metadata_bytes"])}

result = pyarrow.chunked_array(rottnest.rottnest.read_indexed_pages("text", ["s3://redpajama-1t/" + i for i in metadata["filename"].to_list()], 
                                                                    [0] * len(metadata["filename"]),
                                                                    metadata["page_offset_right"].to_list(), 
                                                                    metadata["page_byte_size"].to_list(), 
                                                                    [0] * len(metadata["filename"]), 
                                                                    "aws",
                                                                    file_metadata,
                                                                    False))

print(len(result))
