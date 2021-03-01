import torch
from torch import nn
from torch.nn.parameter import Parameter

from scripts.model import device

# Let be d number of hidden states, m number of premise tokens and n number of hypothesis tokens

# my Stochastic Answer Network implementation
class San(nn.Module):
    def __init__(self, K, n_hidden_states, n_token_P, n_token_H, n_label):
        super(San, self).__init__()
        self.K = K
        self.n_hidden_states = n_hidden_states
        self.n_token_P = n_token_P
        self.n_token_H = n_token_H
        self.n_label = n_label

        self.w1 = Parameter(torch.rand(n_hidden_states, 1)).to(device)
        self.W2 = Parameter(torch.rand(n_hidden_states, n_token_H)).to(device)
        self.W3 = Parameter(torch.rand(n_token_H + n_token_P + 2 * max([n_token_H, n_token_P]), n_label)).to(device)

        self.gru = nn.GRUCell(self.n_token_P, self.n_token_H, bias=False).to(device)

    def forward(self, M_h, M_p):
        P_r = torch.zeros(self.n_label, 1).to(device)
        alpha = nn.functional.softmax(self.w1.t().matmul(M_h),
                                      dim=1)  # alpha: vettore colonna di dimensioni 1 x n (numero token ipotesi)
        sk = self.__compute_k_th(alpha, M_h, (1, self.n_token_H))
        del alpha

        # Loop complexity: O(k) where k is an iperparameter
        for k in range(self.K):
            beta = nn.functional.softmax(sk.matmul(self.W2.t().matmul(M_p)),
                                         dim=1)  # beta: vettore colonna di dimensioni 1 x m (numero token premessa)
            xk = self.__compute_k_th(beta, M_p, (1, self.n_token_P))
            del beta

            sk = self.gru(xk, sk)

            # inserisco gli elementi di s_k, x_k, abs(s_k-x_k) e s_k*x_k per colonna.
            # Ottengo un vettore di dimensione numero_token_premessa+ numero_token_ipotesi+2*max{numero_token_premessa, numero_token_ipotesi}
            max_length = max([self.n_token_H, self.n_token_P])
            sk_1 = torch.cat((sk, torch.zeros(1, max_length - sk.size()[1]).to(device)), 1)
            xk_1 = torch.cat((xk, torch.zeros(1, max_length - xk.size()[1]).to(device)), 1)
            result = torch.cat((sk, xk, torch.abs(sk_1 - xk_1), sk_1 * xk_1), 1).t()
            del sk_1, xk_1
            P_r_k = nn.functional.softmax(self.W3.t().matmul(result), dim=0)
            P_r += P_r_k
            del result, P_r_k
        return (P_r.t() / self.K)

    def __compute_k_th(self, coeff, memory, size_result):
        result = torch.zeros(*size_result).to(device)
        # Loop complexity is O(d)
        for j in range(memory.size()[0]):
            # memory[[j]] vettore riga di dimensioni 1 x m o n (numero token premessa o ipotesi)
            result += coeff[:, [j]].matmul(memory[[j]])
        return result


# this is my class for manage the classification
class Classifier(nn.Module):
    def __init__(self,  K, n_hidden_states, n_token_P, n_token_H, n_label, p):
        super(Classifier, self).__init__()
        self.n_token_P=n_token_P
        self.n_token_H = n_token_H
        self.n_label=n_label
        self.san=San(K, n_hidden_states, n_token_P, n_token_H, n_label).to(device)
        self.dropout = nn.Dropout(p=p)

    def forward(self, last_hidden_state):
        pooled_ouput=torch.empty(0, self.n_label).to(device)
        # Loop complexity: O(h) where h is the bath size, i.e. the number of sequence in the current batch
        for i in range(last_hidden_state.size()[0]):
            M = last_hidden_state[i, :, :].t()
            M = self.dropout(M)
            M_p = M[:, :self.n_token_P]
            M_h = M[:, self.n_token_P:self.n_token_P+self.n_token_H]
            del M
            P_r=self.san(M_h, M_p)
            del M_p, M_h
            pooled_ouput=torch.cat((pooled_ouput, P_r), dim=0)
        return pooled_ouput