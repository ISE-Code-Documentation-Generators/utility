import abc
from typing import Callable, List
from torchtext import vocab

from ise_cdg_utility.metrics.interface import NLPMetricInterface, VectorizedNLPMetric
from ise_cdg_utility.metrics.utils import atomic_map


class VectorizedNLPMetricAdaptor(VectorizedNLPMetric):

    def __init__(self, vocab: vocab.Vocab, nlp_metric: NLPMetricInterface) -> None:
        self.vocab = vocab
        self.nlp_metric = nlp_metric

    def get_itos_translator(self) -> Callable:
        return lambda el: self.vocab.vocab.get_itos()[el]

    def set_references(self, references: List[List[List[int]]]) -> None:
        new_references: List[List[List[str]]] = atomic_map(self.get_itos_translator(), references)
        self.nlp_metric.set_references(new_references)
    
    def __call__(self, candidates: List[List[int]]):
        new_candidates: List[List[str]] = atomic_map(self.get_itos_translator(), candidates)
        return self.nlp_metric(new_candidates)
