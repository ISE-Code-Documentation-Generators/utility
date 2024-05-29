import abc
from typing import List
from torchmetrics.text import BLEUScore, ROUGEScore
from bert_score import score

from ise_cdg_utility.metrics import NLPMetricInterface
from ise_cdg_utility.metrics.src.metric_input_config import (
    CANDIDATES_TYPES,
    REFRENCES_TYPES,
    MetricInputConfigurerStrategy,
)


class NLPMetricTorchmetrics(NLPMetricInterface, abc.ABC):

    def __init__(self) -> None:
        self.input_strategy = None

    @abc.abstractmethod
    def calculate_metric(self, candidates: List[str], references: List[List[str]]):
        pass

    def set_references(self, references: REFRENCES_TYPES) -> None:
        self.input_strategy = MetricInputConfigurerStrategy(references)

    def __call__(self, candidates: CANDIDATES_TYPES):
        assert self.input_strategy is not None, "References is not initialized"
        return self.calculate_metric(*self.input_strategy.configur_inputs(candidates))


class NLPMetricRangedBLEU(NLPMetricTorchmetrics):
    def calculate_metric(self, candidates: List[str], references: List[List[str]]):
        n_gram_limit = 4

        metric_dict = dict()
        for i in range(1, n_gram_limit + 1):
            bleu = BLEUScore(n_gram=i)
            metric_dict[f"bleu_{i}"] = bleu(candidates, references)

        return metric_dict


class NLPMetricROUGE(NLPMetricTorchmetrics):

    def calculate_metric(self, candidates: List[str], references: List[List[str]]):
        rouge = ROUGEScore()
        return rouge(candidates, references)


class NLPMetricBERT(NLPMetricTorchmetrics):
    
    def __init__(self, rescale_with_baseline: bool = True) -> None:
        super().__init__()
        self.use_tqdm = True
        self.rescale_with_baseline = rescale_with_baseline

    def calculate_metric(self, candidates: List[str], references: List[List[str]]):
        return score(candidates, references, model_type="bert-base-uncased", lang="en", rescale_with_baseline=True)
