
from abc import ABC

import torch


class DocumentGeneratorInterface(ABC):

def generate_one_markdown(
        self,
        source,
        source_ast_nodes, source_ast_edges, batch_index,
        start_idx: int,
        device: torch.device,
        eos_ind: int,
        sequence_max_length: int,
):