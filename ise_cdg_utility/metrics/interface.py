import abc
from typing import Dict, List
from torchtext import vocab
from ise_cdg_utility.metrics.enums import CodeMetric


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


def get_vectorized_metrics(vocab: vocab.Vocab) -> Dict[CodeMetric, VectorizedNLPMetric]:
    from ise_cdg_utility.metrics.adaptors import VectorizedNLPMetricAdaptor
    from ise_cdg_utility.metrics.src import (
        NLPMetricRangedBLEU,
        NLPMetricROUGE,
        NLPMetricBERT,
    )

    return {
        CodeMetric.BLEU: VectorizedNLPMetricAdaptor(vocab, NLPMetricRangedBLEU()),
        CodeMetric.ROUGE: VectorizedNLPMetricAdaptor(vocab, NLPMetricROUGE()),
        CodeMetric.BERT: VectorizedNLPMetricAdaptor(vocab, NLPMetricBERT()),
    }


def get_metrics() -> Dict[CodeMetric, NLPMetricInterface]:
    from ise_cdg_utility.metrics.src import (
        NLPMetricRangedBLEU,
        NLPMetricROUGE,
        NLPMetricBERT,
    )

    return {
        CodeMetric.BLEU: NLPMetricRangedBLEU(),
        CodeMetric.ROUGE: NLPMetricROUGE(),
        CodeMetric.BERT: NLPMetricBERT(),
    }


def get_metrics_for_each_result(
    metrics: Dict[CodeMetric, NLPMetricInterface]
) -> Dict[CodeMetric, NLPMetricInterface]:

    from ise_cdg_utility.metrics.src import NLPMetricForEachResult

    new_metrics = {}
    for cm, metric in metrics.items():
        new_metrics[cm] = NLPMetricForEachResult(metric)

    return new_metrics
