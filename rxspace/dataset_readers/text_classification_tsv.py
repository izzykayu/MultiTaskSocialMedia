from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

# You can implement your own dataset reader by subclassing DatasetReader.
# At the very least, you need to implement the _read() method, preferably
# text_to_instance() as well.
@DatasetReader.register("classification_tsv")
class ClassificationTsvReader(DatasetReader):
    """this reads in a text csv with label"""
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, id_: str, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[: self.max_tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields: Dict[str, Field] = {"text": text_field}

        if label:
            fields["label"] = LabelField(label)
        if id_:
            fields["id"] = LabelField(id_)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                tweetid, text, binarylabel = line.strip().split("\t")
                yield self.text_to_instance(tweetid,text, binarylabel)


