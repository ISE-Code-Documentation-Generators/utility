import abc
from typing import List, Union

REFRENCES_TYPES = Union[
    List[List[List[str]]],
    List[List[str]],
    List[str],
]

CANDIDATES_TYPES = Union[
    List[List[str]],
    List[str],
    str,
]


class MetricInputConfigurerStrategy:

    REF_ERR_MSG = "unsupported `references` type"
    INCOMPATIBLE_REF_CAND_ERR_MSG = (
        "the `candidates` type is incompatible with the `references` type"
    )

    @classmethod
    def raise_message_on_not_list(cls, obj):
        if not isinstance(obj, list):
            raise Exception(cls.REF_ERR_MSG)

    def __new__(cls, references: REFRENCES_TYPES):
        cls.raise_message_on_not_list(references)
        if isinstance(references[0], str):
            return super().__new__(MetricInputConfigurerListStrStrategy)

        cls.raise_message_on_not_list(references[0])
        if isinstance(references[0][0], str):
            return super().__new__(MetricInputConfigurerListListStrStrategy)

        cls.raise_message_on_not_list(references[0])
        if isinstance(references[0][0][0], str):
            return super().__new__(MetricInputConfigurerListListListStrStrategy)

    def __init__(self, references: REFRENCES_TYPES) -> None:
        super().__init__()
        self.references = self.configure_references(references)

    @abc.abstractmethod
    def configure_references(self, references: REFRENCES_TYPES) -> List[List[str]]:
        pass

    @abc.abstractmethod
    def configure_candidates(self, candidates: CANDIDATES_TYPES) -> List[str]:
        pass

    def configur_inputs(self, candidates: CANDIDATES_TYPES):
        return self.configure_candidates(candidates), self.references


class MetricInputConfigurerListStrStrategy(MetricInputConfigurerStrategy):

    def configure_references(self, references: List[str]):
        return [references]

    def configure_candidates(self, candidates: str):
        err = False
        try:
            if not isinstance(candidates, str):
                err = True
        except:
            err = True
        finally:
            if err:
                raise Exception(self.INCOMPATIBLE_REF_CAND_ERR_MSG)

        return [candidates]


class MetricInputConfigurerListListStrStrategy(MetricInputConfigurerStrategy):
    def configure_references(self, references: List[List[str]]):
        return references

    def configure_candidates(self, candidates: List[str]):
        err = False
        try:
            if not isinstance(candidates[0], str):
                err = True
        except:
            err = True
        finally:
            if err:
                raise Exception(self.INCOMPATIBLE_REF_CAND_ERR_MSG)

        return candidates


class MetricInputConfigurerListListListStrStrategy(MetricInputConfigurerStrategy):
    def configure_references(self, references: List[List[List[str]]]):
        new_references = []
        for row in references:
            new_row = []
            for refer in row:
                new_row.append(" ".join(refer))
            new_references.append(new_row)
        return new_references

    def configure_candidates(self, candidates: List[List[str]]):
        err = False
        try:
            if not isinstance(candidates[0][0], str):
                err = True
        except:
            err = True
        finally:
            if err:
                raise Exception(self.INCOMPATIBLE_REF_CAND_ERR_MSG)

        new_candidates = []
        for cand in candidates:
            new_candidates.append(" ".join(cand))
        return new_candidates
