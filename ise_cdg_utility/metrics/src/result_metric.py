import typing
from typing import List

from ise_cdg_utility.metrics import NLPMetricInterface

if typing.TYPE_CHECKING:
    from ise_cdg_utility.metrics.src.model_metric import NLPMetricTorchmetrics


class NLPMetricForEachResult(NLPMetricInterface):

    def __init__(self, nlp_metric: "NLPMetricTorchmetrics") -> None:
        super().__init__()
        self.references = None
        self.nlp_metric = nlp_metric

    def _calculate_metric_for_one_candidate(
        self, candidate: List[str], reference: List[List[str]]
    ):
        self.nlp_metric.set_references([reference])
        return self.nlp_metric([candidate])

    def calculate_metric(
        self, candidates: List[List[str]], references: List[List[List[str]]]
    ):
        N = len(candidates)
        result_list = []
        for i in range(N):
            candidate: List[str] = candidates[i]
            reference: List[List[str]] = references[i]
            result_list.append(
                self._calculate_metric_for_one_candidate(candidate, reference)
            )
        return result_list

    def set_references(self, references: List[List[List[str]]]) -> None:
        self.references = references

    def __call__(self, candidates: List[List[str]]):
        assert self.references is not None, "References is not initialized"
        return self.calculate_metric(candidates, self.references)
