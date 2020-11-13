from torch import nn
import torch

from scripts.glue import *

class ObjectiveFunction(nn.Module):
    def __init__(self, obj_task, hidden_size=None, n_classes=None, K=1):
        super(ObjectiveFunction, self).__init__()
        self.task=obj_task
        self.hidden_size=hidden_size
        self.n_classes=n_classes
        self.K=K

    def forward(self, pooled_output):
        linear = nn.Linear(self.hidden_size, self.n_classes, bias=False)
        if isinstance(self.task, SingleSentenceClassification):
            softmax=nn.Softmax(dim=1)
            return softmax(linear(pooled_output))
        elif isinstance(self.task, PairwiseTextClassification):
            softmax=nn.Softmax(dim=1)
            return softmax(linear(pooled_output))
        elif isinstance(self.task, TextSimilarity):
            return linear(pooled_output)
        elif isinstance(self.task, RelevanceRanking):
            sigmoid=nn.Sigmoid()
            return sigmoid(linear(pooled_output))
        else:
            raise TypeError()


