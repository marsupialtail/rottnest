from rottnest.indices.index_interface import RottnestIndex

class SubstringIndex(RottnestIndex):
    def __init__(self, char_index: bool = False, token_skip_factor: int = 1, index_mode: str = 'physical'):
        super().__init__(index_mode)
        self.char_index = char_index
        self.token_skip_factor = token_skip_factor

    def build_index(self, data_arr: pyarrow.Array, uid_arr: pyarrow.Array, output_file_name: str, **kwargs):
        return rottnest.build_lava_substring(output_file_name, data_arr, 
            uid_arr, tokenizer_file, self.token_skip_factor, self.char_index)

    def search_index()