import abc
import typing
from typing import List, Optional
import torch
from torchmetrics.text import BLEUScore, ROUGEScore
from sentence_transformers import SentenceTransformer, util

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

class BERTMetric(NLPMetricTorchmetrics):

    @classmethod
    def calculate_bert_score(self, sentence1, sentence2) -> torch.Tensor:
        model = SentenceTransformer('bert-base-uncased')
        embeddings1 = model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = model.encode(sentence2, convert_to_tensor=True)

        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return cosine_similarity

    def calculate_metric_for_each(self, candidate: str, corresponding_refs: List[str]):
        results = []
        for ref in corresponding_refs:
            results.append(self.calculate_bert_score(candidate, ref))
        combined_tensor = torch.cat(results, dim=0)
        max_value, _ = torch.max(combined_tensor, dim=0)
        return max_value.view(1, 1)

    def calculate_metric(self, candidates: List[str], references: List[List[str]]):
        scores = []
        for i, candidate in enumerate(candidates):
            scores.append(self.calculate_metric_for_each(candidate, references[i]))
        combined_tensor = torch.cat(scores, dim=0)
        average_value = torch.mean(combined_tensor, dim=0)
        return {'bert_score': average_value}
