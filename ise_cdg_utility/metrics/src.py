import abc
import typing
from typing import List, Optional
from torchmetrics.text import BLEUScore, ROUGEScore

from ise_cdg_utility.metrics import NLPMetricInterface


class NLPMetricTorchmetrics(NLPMetricInterface, abc.ABC):

    def __init__(self) -> None:
        self.references = None

    @abc.abstractmethod
    def calculate_metric(self, candidates: List[str], references: List[List[str]]):
        pass

    def set_references(self, references: List[List[List[str]]]) -> None:
        new_references = []
        for row in references:
            new_row = []
            for refer in row:
                new_row.append(' '.join(refer))
            new_references.append(new_row)
        self.references = new_references

    def __call__(self, candidates: List[List[str]]):
        assert self.references is not None, "References is not initialized"
        new_candidates = []
        for cand in candidates:
            new_candidates.append(' '.join(cand))
        return self.calculate_metric(new_candidates, self.references)

class NLPMetricRangedBLEU(NLPMetricTorchmetrics):
    def calculate_metric(self, candidates: List[str], references: List[List[str]]):
        n_gram_limit = 4

        metric_dict = dict()
        for i in range(1, n_gram_limit+1):
            bleu = BLEUScore(n_gram=i)
            metric_dict[f"bleu_{i}"] = bleu(candidates, references) 

        return metric_dict

class NLPMetricROUGE(NLPMetricTorchmetrics):
    
    def calculate_metric(self, candidates: List[str], references: List[List[str]]):
        rouge = ROUGEScore()
        return rouge(candidates, references)

