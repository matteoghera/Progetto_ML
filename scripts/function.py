from torch import nn
import torch

from scripts.glue import *

class ObjectiveFunction:
    def __init__(self, obj_task, hidden_size=None, n_classes=None, K=1):
        self.task=obj_task
        self.hidden_size=hidden_size
        self.n_classes=n_classes
        self.K=K

    def __call__(self, last_hidden_state, pooled_output, embedding_output):
        if isinstance(self.task, SingleSentenceClassification):
            softmax=nn.Softmax(dim=1)
            return softmax(pooled_output)
        elif isinstance(self.task, PairwiseTextClassification):
            batch_size=pooled_output.shape[0]
            softmax = nn.Softmax(dim=1)
            pooled_output = softmax(pooled_output)
            s0 = 0
            for j in range(batch_size):
                s0 += torch.matmul(embedding_output[j, :, :], pooled_output[j, :].reshape(-1, 1))

            pk =torch.tensor([], dtype=torch.double)
            for k in range(self.K):
                sk = s0
                softmax = nn.Softmax(dim=2)
                bj = torch.matmul(sk.reshape(1, -1), last_hidden_state)
                bj = softmax(bj)
                xk = 0
                for j in range(batch_size):
                    xk += torch.matmul(last_hidden_state[j, :, :], bj[j, :, :].reshape(-1, 1))

                xk = torch.reshape(xk, (1, xk.shape[0], xk.shape[1]))
                sk = torch.reshape(sk, (1, sk.shape[0], sk.shape[1]))
                gru = nn.GRU(1, 1, 1)
                sk, w3 = gru(xk, sk)

                matrix_result = torch.cat(
                    (sk[0, :, :], xk[0, :, :], torch.abs(sk[0, :, :] - xk[0, :, :]), sk[0, :, :] - xk[0, :, :]), 1)
                softmax = nn.Softmax(dim=1)
                pk = torch.cat((pk, softmax(torch.matmul(w3[0, :, :].reshape(1, -1), matrix_result))), 1)

            pr = torch.mean(pk, (0, 1))
            return pr
        elif isinstance(self.task, TextSimilarity):
            linear = nn.Linear(self.hidden_size, self.n_classes, bias=False)
            return linear(pooled_output)
        elif isinstance(self.task, RelevanceRanking):
            return self.__get_relevance_ranking_function()
        else:
            raise TypeError()
