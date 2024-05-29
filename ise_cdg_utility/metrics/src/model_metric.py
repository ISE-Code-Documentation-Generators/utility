import abc
from typing import List
import torch
from torchmetrics.text import BLEUScore, ROUGEScore
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

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
    
    def __init__(self) -> None:
        super().__init__()
        self.use_tqdm = True
        self.bert = SentenceTransformer('bert-base-uncased')

    def calculate_bert_score(self, sentence1, sentence2) -> torch.Tensor:
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

        
        embeddings1 = self.bert.encode(sentence1, convert_to_tensor=True)
        embeddings2 = self.bert.encode(sentence2, convert_to_tensor=True)

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
        iterator = enumerate(candidates)
        if self.use_tqdm:
            iterator = tqdm(iterator)

        for i, candidate in iterator:
            scores.append(self.calculate_metric_for_each(candidate, references[i]))
        combined_tensor = torch.cat(scores, dim=0)
        average_value = torch.mean(combined_tensor, dim=0)
        return {"bert_score": average_value}
