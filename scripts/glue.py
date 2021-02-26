import pandas as pd
from path import Path
from torch import float32, nn
import torch
import numpy as np

from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr

from scripts.tokenizer import DatasetPlus
from scripts.san import Classifier


class ObjectiveFunction(nn.Module):
    def __init__(self, obj_task, hidden_size, n_classes):
        super(ObjectiveFunction, self).__init__()
        self.task=obj_task
        self.linear = nn.Linear(hidden_size, n_classes, bias=False)

        n_token_P, n_token_H = self.task.compute_sentences_statistic()
        if n_token_P is not None and n_token_H is not None:
            self.classifier=Classifier(50, hidden_size, n_token_P, n_token_H, n_classes, self.task.get_dropout_parameter())
        else:
            self.classifier=None

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid=nn.Sigmoid()


    def forward(self, last_hidden_state, pooled_output):
        pooled_output = self.linear(pooled_output)
        if isinstance(self.task, ClassificationTask):
            if self.classifier is None:
                pooled_output=self.softmax(pooled_output)
                return pooled_output
            else:
                pooled_output=self.classifier(last_hidden_state)
                return pooled_output
        elif isinstance(self.task, TextSimilarity):
            return pooled_output
        elif isinstance(self.task, RelevanceRanking):
            pooled_output=self.sigmoid(pooled_output)
            return pooled_output
        else:
            raise TypeError()

MAX_ROWS_TRAIN=10000
MAX_ROWS_EVAL=8000

class Tasks:
    TOTAL_BATCH = 0
    def __init__(self, path):
        self.path = path

    def load_dev_test_train(self):
        self.dev = pd.read_csv(self.path / "dev.tsv", nrows=MAX_ROWS_EVAL, sep="\t", error_bad_lines=False)
        self.train = pd.read_csv(self.path / "train.tsv", nrows=MAX_ROWS_TRAIN, sep="\t", error_bad_lines=False)
        self.test = pd.read_csv(self.path / "test.tsv", sep="\t", error_bad_lines=False)

    def data_cleanup(self):
        pass

    def get_dropout_parameter(self):
        return 0.1

    def get_name(self):
        pass

    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        pass

    def get_dev(self):
        pass

    def get_train(self):
        pass

    def get_test(self):
        pass

    def get_objective_function(self, hidden_size):
        pass

    def get_loss_function(self):
        pass

    def predict(self, pooled_output):
        pass

    def print_metrics(self, loss, acc, phase):
        if phase.__eq__("train"):
            print(f'Train Cross Entropy {loss} accuracy {acc}')
        else:
            print(f'Validation Cross Entropy {loss} accuracy {acc}')

    def compute_matric_value(self, preds, targets, n_examples):
        return accuracy_score(targets,preds)*(len(targets)/n_examples)

    def compute_sentences_statistic(self):
        return None, None

class ClassificationTask(Tasks):
    def __init__(self, path):
        super().__init__(path)

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def predict(self, pooled_output):
        _, preds = torch.max(pooled_output, dim=1)
        return preds

class SingleSentenceClassification(ClassificationTask):
    def __init__(self, path):
        super().__init__(path)

    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        self.dev_tokenized_data = DatasetPlus(self.dev, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence",
                                   column_target="label")

        self.train_tokenized_data = DatasetPlus(self.train, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence",
                                   column_target="label")

        self.test_tokenized_data = DatasetPlus(self.test, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence")
        Tasks.TOTAL_BATCH=Tasks.TOTAL_BATCH+len(self.train_tokenized_data.get_dataloader())

    def get_dev(self):
        return self.dev, self.dev_tokenized_data.get_dataloader()

    def get_train(self):
        return self.train, self.train_tokenized_data.get_dataloader()

    def get_test(self):
        return self.test, self.test_tokenized_data.get_dataloader()

    def get_objective_function(self, hidden_size):
        return ObjectiveFunction(self, hidden_size, n_classes=self.dev["label"].nunique())


class PairwiseTextClassification(ClassificationTask):
    def __init__(self, path):
        super().__init__(path)

    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        self.dev_tokenized_data = DatasetPlus(self.dev, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2", column_target="label")

        self.train_tokenized_data = DatasetPlus(self.train, tokenizer, max_len, batch_size, num_workers,
                                   column_sequence1="sentence1", column_sequence2="sentence2", column_target="label")

        self.test_tokenized_data = DatasetPlus(self.test, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2")
        Tasks.TOTAL_BATCH = Tasks.TOTAL_BATCH + len(self.train_tokenized_data.get_dataloader())
    def get_dev(self):
        return self.dev, self.dev_tokenized_data.get_dataloader()

    def get_train(self):
        return self.train, self.train_tokenized_data.get_dataloader()

    def get_test(self):
        return self.test, self.test_tokenized_data.get_dataloader()

    def get_objective_function(self, hidden_size):
        return ObjectiveFunction(self, hidden_size, n_classes=self.dev["label"].nunique())

    def compute_sentences_statistic(self):
        train_dataloader_iterator=self.train_tokenized_data.viewData()
        list_n_token_P, list_n_token_H=self.__compute_number_token_sentences(train_dataloader_iterator)

        dev_dataloader_iterator = self.dev_tokenized_data.viewData()
        list_n_token_P_1, list_n_token_H_1 = self.__compute_number_token_sentences(dev_dataloader_iterator)

        list_n_token_P.extend(list_n_token_P_1)
        list_n_token_H.extend(list_n_token_H_1)

        list_n_token_P=np.array(list_n_token_P)
        list_n_token_H = np.array(list_n_token_H)
        return np.ceil(np.mean(list_n_token_P)), np.ceil(np.mean(list_n_token_H))

    def __compute_number_token_sentences(self, train_dataloader_iterator):
        stop = False
        list_n_token_P = []
        list_n_token_H = []
        while not stop:
            data = next(train_dataloader_iterator, None)
            if data is not None:
                for i in range(len(data["sequence1"])):
                    result = np.array([data['attention_mask'].tolist()[i],
                                       data["token_type_ids"].tolist()[i]])
                    temp = np.sum(result, axis=0)
                    n_token_P = sum(filter(lambda e: e == 1, temp))
                    n_token_H = sum(filter(lambda e: e == 2, temp))
                    list_n_token_P.append(n_token_P)
                    list_n_token_H.append(n_token_H)
            else:
                stop = True
        return list_n_token_P, list_n_token_H


class TextSimilarity(Tasks):
    def __init__(self, path):
        super().__init__(path)

    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        self.dev_tokenized_data = DatasetPlus(self.dev, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2", column_target="score", dtype=float32)

        self.train_tokenized_data = DatasetPlus(self.train, tokenizer, max_len, batch_size, num_workers,
                                   column_sequence1="sentence1", column_sequence2="sentence2", column_target="score", dtype=float32)

        self.test_tokenized_data = DatasetPlus(self.test, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2", dtype=float32)
        Tasks.TOTAL_BATCH = Tasks.TOTAL_BATCH + len(self.train_tokenized_data.get_dataloader())
    def get_dev(self):
        return self.dev, self.dev_tokenized_data.get_dataloader()

    def get_train(self):
        return self.train, self.train_tokenized_data.get_dataloader()

    def get_test(self):
        return self.test, self.test_tokenized_data.get_dataloader()

    def get_objective_function(self, hidden_size):
        return ObjectiveFunction(self, hidden_size, n_classes=1)

    def get_loss_function(self):
        return self.TextSimilarityLoss()

    class TextSimilarityLoss(nn.Module):
        def __init__(self):
            super(TextSimilarity.TextSimilarityLoss, self).__init__()
            self.loss=nn.MSELoss()

        def forward(self, input, target):
            target=torch.reshape(target, (target.shape[0], 1))
            return self.loss(input, target)

    def predict(self, pooled_output):
        return torch.reshape(pooled_output, (-1,))

class RelevanceRanking(Tasks):
    def __init__(self, path):
        super().__init__(path)

    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        self.dev_tokenized_data = DatasetPlus(self.dev, tokenizer, max_len, batch_size, num_workers, column_sequence1="question",
                                   column_sequence2="sentence", column_target="label_encoding")

        self.train_tokenized_data = DatasetPlus(self.train, tokenizer, max_len, batch_size, num_workers,
                                   column_sequence1="question", column_sequence2="sentence",
                                   column_target="label_encoding")

        self.test_tokenized_data = DatasetPlus(self.test, tokenizer, max_len, batch_size, num_workers, column_sequence1="question",
                                   column_sequence2="sentence")
        Tasks.TOTAL_BATCH = Tasks.TOTAL_BATCH + len(self.train_tokenized_data.get_dataloader())

    def get_dev(self):
        return self.dev, self.dev_tokenized_data.get_dataloader()

    def get_train(self):
        return self.train, self.train_tokenized_data.get_dataloader()

    def get_test(self):
        return self.test, self.test_tokenized_data.get_dataloader()

    def get_objective_function(self, hidden_size):
        return ObjectiveFunction(self, hidden_size, n_classes=self.dev["label_encoding"].nunique())

    def get_loss_function(self):
        return nn.NLLLoss()

    def predict(self, pooled_output):
        _, preds = torch.max(pooled_output, dim=1)
        return preds


### Single-Sentence Classification tasks

class CoLA(SingleSentenceClassification):
    def __init__(self, path):
        super().__init__(path / "CoLA")
        self.load_dev_test_train()
        self.data_cleanup()

    def load_dev_test_train(self):
        self.dev = pd.read_csv(self.path / "dev.tsv", sep="\t", nrows=MAX_ROWS_EVAL, header=None, error_bad_lines=False)
        self.train = pd.read_csv(self.path / "train.tsv", sep="\t", nrows=MAX_ROWS_TRAIN, header=None, error_bad_lines=False)
        self.test = pd.read_csv(self.path / "test.tsv", sep="\t", error_bad_lines=False)

    def data_cleanup(self):
        self.dev.drop(labels=2, axis=1, inplace=True)
        self.dev.drop(labels=0, axis=1, inplace=True)
        self.dev.rename(columns={1: "label", 3: "sentence"}, inplace=True)

        self.train.drop(labels=2, axis=1, inplace=True)
        self.train.drop(labels=0, axis=1, inplace=True)
        self.train.rename(columns={1: "label", 3: "sentence"}, inplace=True)

        self.test.drop(labels="index", axis=1, inplace=True)

    def get_dropout_parameter(self):
        return 0.05

    def get_name(self):
        return "CoLA"

    def print_metrics(self, loss, acc, phase):
        if phase.__eq__("train"):
            print(f'Train Cross Entropy {loss} Matthews corr. {acc}')
        else:
            print(f'Validation Cross Entropy {loss} Matthews corr. {acc}')

    def compute_matric_value(self, preds, targets, n_examples):
        return matthews_corrcoef(targets, preds)

class SST_2(SingleSentenceClassification):
    def __init__(self, path):
        super().__init__(path / "SST-2")
        self.load_dev_test_train()
        self.data_cleanup()

    def data_cleanup(self):
        self.test.drop(labels="index", axis=1, inplace=True)

    def get_name(self):
        return "SST-2"

### Pairwise Text Classification tasks

class RTE(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "RTE")
        self.load_dev_test_train()
        self.data_cleanup()

    def data_cleanup(self):
        self.train.drop(labels="index", axis=1, inplace=True)
        self.dev.drop(labels="index", axis=1, inplace=True)
        self.test.drop(labels="index", axis=1, inplace=True)

        self.train["label_encoding"] = self.train["label"].map({"not_entailment": 0, "entailment": 1})
        self.dev["label_encoding"] = self.dev["label"].map({"not_entailment": 0, "entailment": 1})

        self.train.dropna(inplace=True)

    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        self.dev_tokenized_data = DatasetPlus(self.dev, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2", column_target="label_encoding")

        self.train_tokenized_data = DatasetPlus(self.train, tokenizer, max_len, batch_size, num_workers,
                                   column_sequence1="sentence1", column_sequence2="sentence2",
                                   column_target="label_encoding")

        self.test_tokenized_data = DatasetPlus(self.test, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2")
        Tasks.TOTAL_BATCH = Tasks.TOTAL_BATCH + len(self.train_tokenized_data.get_dataloader())

    def get_objective_function(self, hidden_size):
        return ObjectiveFunction(self, hidden_size=hidden_size, n_classes=self.dev["label_encoding"].nunique())

    def get_name(self):
        return "RTE"

class WNLI(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "WNLI")
        self.load_dev_test_train()
        self.data_cleanup()

    def data_cleanup(self):
        self.train.drop(labels="index", axis=1, inplace=True)
        self.dev.drop(labels="index", axis=1, inplace=True)
        self.test.drop(labels="index", axis=1, inplace=True)

    def get_name(self):
        return "WNLI"


class QQP(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "QQP")
        self.load_dev_test_train()
        self.data_cleanup()

    def data_cleanup(self):
        cols_id = range(3)
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)
        self.test.drop(labels="id", axis=1, inplace=True)

    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        self.dev_tokenized_data  = DatasetPlus(self.dev, tokenizer, max_len, batch_size, num_workers, column_sequence1="question1",
                                   column_sequence2="question2", column_target="is_duplicate")

        self.train_tokenized_data = DatasetPlus(self.train, tokenizer, max_len, batch_size, num_workers,
                                   column_sequence1="question1", column_sequence2="question2",
                                   column_target="is_duplicate")

        self.test_tokenized_data = DatasetPlus(self.test, tokenizer, max_len, batch_size, num_workers, column_sequence1="question1",
                                   column_sequence2="question2")
        Tasks.TOTAL_BATCH = Tasks.TOTAL_BATCH + len(self.train_tokenized_data.get_dataloader())

    def get_objective_function(self, hidden_size):
        return ObjectiveFunction(self, hidden_size=hidden_size, n_classes=self.dev["is_duplicate"].nunique())

    def get_name(self):
        return "QQP"


class MRPC(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "MRPC")
        self.load_dev_test_train()
        self.data_cleanup()

    def load_dev_test_train(self):
        super().load_dev_test_train()
        # self.dev_ids=pd.read_csv(self.path / "dev_ids.tsv", sep="\t", header=None, error_bad_lines=False)
        #self.msr_paraphrase_test = pd.read_csv(self.path / "msr_paraphrase_test.txt", sep="\t", error_bad_lines=False)
        # self.msr_paraphrase_train = pd.read_csv(self.path / "msr_paraphrase_train.txt", sep="\t", error_bad_lines=False)

    def data_cleanup(self):
        cols_id = [1, 2]
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.dev.rename(columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"}, inplace=True)
        #self.msr_paraphrase_test.drop(columns=self.msr_paraphrase_test.columns[cols_id], inplace=True)
        #self.msr_paraphrase_test.rename(
        #    columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"}, inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)
        self.train.rename(columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"},
                          inplace=True)
        cols_id = range(3)
        self.test.drop(columns=self.test.columns[cols_id], inplace=True)
        self.test.rename(columns={"#1 String": "sentence1", "#2 String": "sentence2"}, inplace=True)

        self.train.dropna(inplace=True)


    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        super().tokenization(tokenizer, max_len, batch_size, num_workers)
        #self.msr_paraphrase_tokenized_data = DatasetPlus(self.msr_paraphrase_test, tokenizer, max_len, batch_size, num_workers,
        #                           column_sequence1="sentence1", column_sequence2="sentence2", column_target="label")

    def get_name(self):
        return "MRPC"


class SNLI(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "SNLI")
        self.load_dev_test_train()
        self.data_cleanup()

    def load_dev_test_train(self):
        super().load_dev_test_train()
        # self.dev_jsonl=self.__from_json("dev.jsonl")
        # self.test_jsonl = self.__from_json("test.jsonl")
        # self.train_jsonl = self.__from_json("train.jsonl")

    def data_cleanup(self):
        cols_id = range(7)
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.test.drop(columns=self.test.columns[cols_id], inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)

        self.dev["gold_label_encoding"] = self.dev["gold_label"].map(
            {"neutral": 0, "contradiction": 1, "entailment": 2})
        self.train["gold_label_encoding"] = self.train["gold_label"].map(
            {"neutral": 0, "contradiction": 1, "entailment": 2})

        self.train.dropna(inplace=True)

    def __from_json(self, file_name):
        import re
        import ast
        with open(self.path / file_name) as f:
            pattern = '"annotator_labels":\s.("neutral",\s|"entailment",\s|"contradiction",\s|"neutral"|"entailment"|"contradiction")*.,\s'
            data = f.read()
            data = re.sub(pattern, "", data)
            data = data.split(sep="\n")
            data = [ast.literal_eval(row) for row in data[:9999]]

            data = pd.DataFrame(data)
        return data

    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        self.dev_tokenized_data = DatasetPlus(self.dev, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2", column_target="gold_label_encoding")

        self.train_tokenized_data = DatasetPlus(self.train, tokenizer, max_len, batch_size, num_workers,
                                   column_sequence1="sentence1", column_sequence2="sentence2",
                                   column_target="gold_label_encoding")

        self.test_tokenized_data = DatasetPlus(self.test, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2")
        Tasks.TOTAL_BATCH = Tasks.TOTAL_BATCH + len(self.train_tokenized_data.get_dataloader())

    def get_objective_function(self, hidden_size):
        return ObjectiveFunction(self, hidden_size=hidden_size, n_classes=self.dev["gold_label_encoding"].nunique())

    def get_name(self):
        return "SNLI"


class MNLI(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "MNLI")
        self.load_dev_test_train()
        self.data_cleanup()

    def load_dev_test_train(self):
        dev_matched = pd.read_csv(self.path / "dev_matched.tsv", sep="\t", nrows=MAX_ROWS_EVAL, error_bad_lines=False)
        dev_mismatched = pd.read_csv(self.path / "dev_mismatched.tsv", sep="\t", nrows=MAX_ROWS_TRAIN, error_bad_lines=False)
        self.dev = pd.concat([dev_matched, dev_mismatched], ignore_index=True)

        test_matched = pd.read_csv(self.path / "test_matched.tsv", sep="\t", error_bad_lines=False)
        test_mismatched = pd.read_csv(self.path / "test_mismatched.tsv", sep="\t", error_bad_lines=False)
        self.test = pd.concat([test_matched, test_mismatched], ignore_index=True)

        self.train = pd.read_csv(self.path / "train.tsv", sep="\t", error_bad_lines=False)

        # self.diagnostic = pd.read_csv(self.path / "diagnostic.tsv", sep="\t", error_bad_lines=False)
        #self.diagnostic_full = pd.read_csv(self.path / "diagnostic-full.tsv", sep="\t",
        #                                   error_bad_lines=False)  # is not # in the table 3.1


    def tokenization(self, tokenizer, max_len, batch_size, num_workers):
        self.dev_tokenized_data = DatasetPlus(self.dev, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2", column_target="gold_label_encoding")

        self.train_tokenized_data = DatasetPlus(self.train, tokenizer, max_len, batch_size, num_workers,
                                   column_sequence1="sentence1", column_sequence2="sentence2",
                                   column_target="gold_label_encoding")

        self.test_tokenized_data = DatasetPlus(self.test, tokenizer, max_len, batch_size, num_workers, column_sequence1="sentence1",
                                   column_sequence2="sentence2")
        Tasks.TOTAL_BATCH = Tasks.TOTAL_BATCH + len(self.train_tokenized_data.get_dataloader())

    def data_cleanup(self):
        cols_id = range(8)
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.test.drop(columns=self.test.columns[cols_id], inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)
        #cols_id = range(5)
        #self.diagnostic_full.drop(columns=self.diagnostic_full.columns[cols_id], inplace=True)

        self.dev["gold_label_encoding"] = self.dev["gold_label"].map(
            {"neutral": 0, "contradiction": 1, "entailment": 2})
        #self.diagnostic_full["label_encoding"] = self.diagnostic_full["Label"].map(
        #    {"neutral": 0, "contradiction": 1, "entailment": 2})
        self.train["gold_label_encoding"] = self.train["gold_label"].map(
            {"neutral": 0, "contradiction": 1, "entailment": 2})

        self.train.dropna(inplace=True)

    def get_objective_function(self, hidden_size):
        return ObjectiveFunction(self, hidden_size=hidden_size, n_classes=self.dev["gold_label_encoding"].nunique())

    def get_dropout_parameter(self):
        return 0.3

    def get_name(self):
        return "MNLI"


### Text Similarity tasks

class STS_B(TextSimilarity):
    def __init__(self, path):
        super().__init__(path / "STS-B")
        self.load_dev_test_train()
        self.data_cleanup()

    def load_dev_test_train(self):
        import csv
        self.dev = pd.read_csv(self.path / "dev.tsv", sep="\t", nrows=MAX_ROWS_EVAL, error_bad_lines=False)
        self.train = pd.read_csv(self.path / "train.tsv", sep="\t", nrows=MAX_ROWS_TRAIN, error_bad_lines=False)
        self.test = pd.read_csv(self.path / "test.tsv", sep="\t", error_bad_lines=False,
                                quoting=csv.QUOTE_NONE)  # engine='python')

    def data_cleanup(self):
        cols_id = range(7)
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.test.drop(columns=self.test.columns[cols_id], inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)

        self.dev.dropna(inplace=True)
        self.test.dropna(inplace=True)
        self.train.dropna(inplace=True)


    def get_name(self):
        return "STS-B"

    def print_metrics(self, loss, acc, phase):
        if phase.__eq__("train"):
            print(f'Train MSE {loss} pearson {acc}')
        else:
            print(f'Validation MSE {loss} pearson {acc}')

    def compute_matric_value(self,preds, targets, n_examples):
        r, _=pearsonr(preds, targets)
        return r


### Relevance Ranking tasks

class QNLI(RelevanceRanking):
    def __init__(self, path):
        super().__init__(path / "QNLI")
        self.load_dev_test_train()
        self.data_cleanup()

    def data_cleanup(self):
        self.train.drop(labels="index", axis=1, inplace=True)
        self.dev.drop(labels="index", axis=1, inplace=True)
        self.test.drop(labels="index", axis=1, inplace=True)

        self.train["label_encoding"] = self.train["label"].map({"not_entailment": 0, "entailment": 1})
        self.dev["label_encoding"] = self.dev["label"].map({"not_entailment": 0, "entailment": 1})

    def get_name(self):
        return "QNLI"

