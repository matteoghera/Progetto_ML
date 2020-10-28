import pandas as pd
from path import Path

class Tasks:
    def __init__(self, path):
        self.path=path

    def load_dev_test_train(self):
        self.dev=pd.read_csv(self.path / "dev.tsv", sep="\t", error_bad_lines=False)
        self.train=pd.read_csv(self.path / "train.tsv",sep="\t", error_bad_lines=False)
        self.test=pd.read_csv(self.path/ "test.tsv", sep="\t", error_bad_lines=False)

class ClassificationTask(Tasks):
    def __init__(self, path):
        super().__init__(path)

class SingleSentenceClassification(ClassificationTask):
    def __init__(self, path):
        super().__init__(path)

class PairwiseTextClassification(ClassificationTask):
    def __init__(self, path):
        super().__init__(path)

class TextSimilarity(Tasks):
    def __init__(self, path):
        super().__init__(path)

class RelevanceRanking(Tasks):
    def __init__(self, path):
        super().__init__(path)


### Single-Sentence Classification tasks

class CoLA(SingleSentenceClassification):
    def __init__(self, path):
        super().__init__(path / "CoLA")
        self.load_dev_test_train()
        self.__data_cleanup()

    def load_dev_test_train(self):
        self.dev = pd.read_csv(self.path / "dev.tsv", sep="\t", header=None, error_bad_lines=False)
        self.train = pd.read_csv(self.path / "train.tsv", sep="\t", header=None, error_bad_lines=False)
        self.test = pd.read_csv(self.path / "test.tsv", sep="\t", error_bad_lines=False)

    def __data_cleanup(self):
        self.dev.drop(labels=2, axis=1, inplace=True)
        self.dev.drop(labels=0, axis=1, inplace=True)
        self.dev.rename(columns={1: "label", 3: "sentence"}, inplace=True)

        self.train.drop(labels=2, axis=1, inplace=True)
        self.train.drop(labels=0, axis=1, inplace=True)
        self.train.rename(columns={1: "label", 3: "sentence"}, inplace=True)

        self.test.drop(labels="index", axis=1, inplace=True)

class SST_2(SingleSentenceClassification):
    def __init__(self, path):
        super().__init__(path / "SST-2")
        self.load_dev_test_train()
        self.__data_cleanup()

    def __data_cleanup(self):
        self.test.drop(labels="index", axis=1, inplace=True)


### Pairwise Text Classification tasks

class RTE(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "RTE")
        self.load_dev_test_train()
        self.__data_cleanup()

    def __data_cleanup(self):
        self.train.drop(labels="index", axis=1, inplace=True)
        self.dev.drop(labels="index", axis=1, inplace=True)
        self.test.drop(labels="index", axis=1, inplace=True)

class WNLI(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "WNLI")
        self.load_dev_test_train()
        self.__data_cleanup()

    def __data_cleanup(self):
        self.train.drop(labels="index", axis=1, inplace=True)
        self.dev.drop(labels="index", axis=1, inplace=True)
        self.test.drop(labels="index", axis=1, inplace=True)

class QQP(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "QQP")
        self.load_dev_test_train()
        self.__data_cleanup()

    def __data_cleanup(self):
        cols_id=range(3)
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)
        self.test.drop(labels="id", axis=1, inplace=True)


class MRPC(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "MRPC")
        self.load_dev_test_train()
        #self.dev_ids=pd.read_csv(self.path / "dev_ids.tsv", sep="\t", header=None, error_bad_lines=False)
        self.msr_paraphrase_test=pd.read_csv(self.path / "msr_paraphrase_test.txt", sep="\t", error_bad_lines=False)
        #self.msr_paraphrase_train = pd.read_csv(self.path / "msr_paraphrase_train.txt", sep="\t", error_bad_lines=False)
        self.__data_cleanup()

    def __data_cleanup(self):
        cols_id=[1,2]
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.dev.rename(columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"}, inplace=True)
        self.msr_paraphrase_test.drop(columns=self.msr_paraphrase_test.columns[cols_id], inplace=True)
        self.msr_paraphrase_test.rename(columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"}, inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)
        self.train.rename(columns={"Quality": "label", "#1 String": "sentence1", "#2 String": "sentence2"}, inplace=True)
        cols_id=range(3)
        self.test.drop(columns=self.test.columns[cols_id], inplace=True)
        self.test.rename(columns={"#1 String": "sentence1", "#2 String": "sentence2"}, inplace=True)

class SNLI(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "SNLI")
        self.load_dev_test_train()
        #self.dev_jsonl=self.__from_json("dev.jsonl")
        #self.test_jsonl = self.__from_json("test.jsonl")
        #self.train_jsonl = self.__from_json("train.jsonl")
        self.__data_cleanup()

    def __data_cleanup(self):
        cols_id = range(7)
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.test.drop(columns=self.test.columns[cols_id], inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)

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

class MNLI(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "MNLI")
        dev_matched=pd.read_csv(self.path / "dev_matched.tsv", sep="\t", error_bad_lines=False)
        dev_mismatched = pd.read_csv(self.path / "dev_mismatched.tsv", sep="\t", error_bad_lines=False)
        self.dev=pd.concat([dev_matched,dev_mismatched], ignore_index=True)

        test_matched=pd.read_csv(self.path / "test_matched.tsv", sep="\t", error_bad_lines=False)
        test_mismatched = pd.read_csv(self.path / "test_mismatched.tsv", sep="\t", error_bad_lines=False)
        self.test = pd.concat([test_matched, test_mismatched], ignore_index=True)

        self.train = pd.read_csv(self.path / "train.tsv", sep="\t", error_bad_lines=False)

        #self.diagnostic = pd.read_csv(self.path / "diagnostic.tsv", sep="\t", error_bad_lines=False)
        self.diagnostic_full = pd.read_csv(self.path / "diagnostic-full.tsv", sep="\t", error_bad_lines=False) # is not in the table 3.1

        self.__data_cleanup()

    def __data_cleanup(self):
        cols_id = range(8)
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.test.drop(columns=self.test.columns[cols_id], inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)
        cols_id = range(5)
        self.diagnostic_full.drop(columns=self.diagnostic_full.columns[cols_id], inplace=True)


### Text Similarity tasks

class STS_B(TextSimilarity):
    def __init__(self, path):
        super().__init__(path / "STS-B")
        self.load_dev_test_train()
        self.__data_cleanup()

    def load_dev_test_train(self):
        import csv
        self.dev=pd.read_csv(self.path / "dev.tsv", sep="\t", error_bad_lines=False)
        self.train=pd.read_csv(self.path / "train.tsv",sep="\t", error_bad_lines=False)
        self.test=pd.read_csv(self.path/ "test.tsv", sep="\t", error_bad_lines=False, quoting=csv.QUOTE_NONE) #engine='python')

    def __data_cleanup(self):
        cols_id=range(7)
        self.dev.drop(columns=self.dev.columns[cols_id], inplace=True)
        self.test.drop(columns=self.test.columns[cols_id], inplace=True)
        self.train.drop(columns=self.train.columns[cols_id], inplace=True)


### Relevance Ranking tasks

class QNLI(RelevanceRanking):
    def __init__(self, path):
        super().__init__(path / "QNLI")
        self.load_dev_test_train()
        self.__data_cleanup()

    def __data_cleanup(self):
        self.train.drop(labels="index", axis=1, inplace=True)
        self.dev.drop(labels="index", axis=1, inplace=True)
        self.test.drop(labels="index", axis=1, inplace=True)