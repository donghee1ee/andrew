from typing import Optional
from dataclasses import dataclass

import torch
from transformers.file_utils import ModelOutput

@dataclass
class RegressionModelOutput(ModelOutput):
    """
    Base class for model's outputs that includes loss and predictions.
    """
    predictions: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None