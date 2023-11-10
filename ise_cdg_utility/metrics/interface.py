import abc
from typing import Callable, List, Tuple
from torchtext import vocab

from ise_cdg_utility.metrics.utils import atomic_map



class NLPMetricInterface(abc.ABC):

    @abc.abstractmethod
    def set_references(self, references: List[List[List[str]]]) -> None:
        pass
        

    @abc.abstractmethod
    def __call__(self, candidates: List[List[str]]):
        pass


class VectorizedNLPMetric(abc.ABC):
    vocab: vocab.Vocab

    @abc.abstractmethod
    def set_references(self, references: List[List[List[int]]]) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, candidates: List[List[int]]):
        pass


def get_vectorized_metrics(vocab: vocab.Vocab) -> Tuple[VectorizedNLPMetric, VectorizedNLPMetric]:
    from ise_cdg_utility.metrics.adaptors import VectorizedNLPMetricAdaptor
    from ise_cdg_utility.metrics.src import NLPMetricRangedBLEU, NLPMetricROUGE

    return VectorizedNLPMetricAdaptor(vocab, NLPMetricRangedBLEU()), VectorizedNLPMetricAdaptor(vocab, NLPMetricROUGE())
