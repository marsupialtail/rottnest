import pyarrow
import polars
import rottnest_rs

a = polars.from_dict({"a":["你是一只小猪","hello you are happy", "hello, et tu, brutes?"]}).to_arrow()
b = polars.from_dict({"a":["cn","en", "bump"]}).to_arrow()
text = a["a"].combine_chunks().cast(pyarrow.string())
uid = pyarrow.array([1,2,3]).cast(pyarrow.uint64())
language = b["a"].combine_chunks().cast(pyarrow.string())
print(rottnest_rs.tokenize_natural_language(text, uid, language))
print(rottnest_rs.search_lava("output_file.bin", "猪"))
