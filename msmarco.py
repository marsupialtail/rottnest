import rottnest
import os
# for i in range(1,3):
#     rottnest.index_file_bm25(f"msmarco/chunk_{i}.parquet","body", name = f"msmarco_index/{i}")
# rottnest.merge_index_bm25("msmarco_index/merged", [f"msmarco_index/{i}" for i in range(1,3)])

# rottnest.merge_index_bm25("msmarco_index/merged", [f"../index/condensed_{i}" for i in range(10)])

# rottnest.index_file_substring(f"msmarco/chunk_1.parquet","body", name = f"msmarco_index/test")
# result = rottnest.search_index_bm25(["msmarco_index/merged"], "politics", K = 10,query_expansion = "openai")
# print(result)

# for i, f in enumerate([f for f in os.listdir("../local") if f.endswith('.parquet')]):
#    rottnest.index_file_bm25(f"../local/{f}", "content_split", name = f"chinese_index/{i}", tokenizer_file = "../tok/tokenizer.json")
# rottnest.merge_index_bm25("local_index/merged", [f"local_index/{f}" for f in os.listdir("../local") if f.endswith('.parquet')])
    
# for i, f in enumerate([f for f in os.listdir("../local") if f.endswith('.parquet')]):
# for i, f in enumerate([f"../local/part-000{str(i).zfill(2)}-78eda15d-705a-4698-855b-0b9864ca355a-c000.gz.parquet" for i in range(20)]):
#     print(f)
#     rottnest.index_file_substring(f"{f}", "content_split", name = f"chinese_index/{i}", tokenizer_file = "../tok/tokenizer.json")
#     print(rottnest.search_index_substring([f"chinese_index/{i}"], "iPhone 13 Pro", 10))

# rottnest.merge_index_substring("chinese_index/condensed", [f"chinese_index/{i}" for i in range(5, 20)])
# print(rottnest.search_index_substring(["chinese_index/condensed"], " Joe Biden", 10))

import h5py
test_vector = h5py.File("mnist/mnist.hdf5")["test"][23]
print(h5py.File("mnist/mnist.hdf5")["neighbors"][23][:10])
# rottnest.index_file_vector("mnist/test.parquet", "vector", name = "mnist/mnist_index")
# rottnest.search_index_vector(["mnist/mnist_index", "mnist/mnist_index"], test_vector, 10)

# rottnest.index_file_vector("mnist/shard0.parquet", "vector", name = "mnist/mnist_index0")
# rottnest.index_file_vector("mnist/shard1.parquet", "vector", name = "mnist/mnist_index1")
rottnest.merge_index_vector("mnist/mnist_index2", ["mnist/mnist_index0", "mnist/mnist_index1"])
# rottnest.search_index_vector(["mnist/mnist_index0", "mnist/mnist_index1"], test_vector, 10)
rottnest.search_index_vector(["mnist/mnist_index2"], test_vector, 10)