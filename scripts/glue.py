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

    def load_dev_test_train(self):
        self.dev = pd.read_csv(self.path / "dev.tsv", sep="\t", header=None, error_bad_lines=False)
        self.train = pd.read_csv(self.path / "train.tsv", sep="\t", header=None, error_bad_lines=False)
        self.test = pd.read_csv(self.path / "test.tsv", sep="\t", error_bad_lines=False)

class SST_2(SingleSentenceClassification):
    def __init__(self, path):
        super().__init__(path / "SST-2")
        self.load_dev_test_train()


### Pairwise Text Classification tasks

class RTE(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "RTE")
        self.load_dev_test_train()

class WNLI(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "WNLI")
        self.load_dev_test_train()

class QQP(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "QQP")
        self.load_dev_test_train()

class MRPC(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "MRPC")
        self.load_dev_test_train()
        self.dev_ids=pd.read_csv(self.path / "dev_ids.tsv", sep="\t", header=None, error_bad_lines=False)
        self.msr_paraphrase_test=pd.read_csv(self.path / "msr_paraphrase_test.txt", sep="\t", error_bad_lines=False)
        self.msr_paraphrase_test = pd.read_csv(self.path / "msr_paraphrase_train.txt", sep="\t", error_bad_lines=False)

class SNLI(PairwiseTextClassification):
    def __init__(self, path):
        super().__init__(path / "SNLI")
        self.load_dev_test_train()
        self.dev_jsonl=self.__from_json("dev.jsonl")
        self.test_jsonl = self.__from_json("test.jsonl")
        self.train_jsonl = self.__from_json("train.jsonl")

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
        self.dev_matched=pd.read_csv(self.path / "dev_matched.tsv", sep="\t", error_bad_lines=False)
        self.dev_mismatched = pd.read_csv(self.path / "dev_mismatched.tsv", sep="\t", error_bad_lines=False)
        self.diagnostic = pd.read_csv(self.path / "diagnostic.tsv", sep="\t", error_bad_lines=False)
        self.diagnostic_full = pd.read_csv(self.path / "diagnostic-full.tsv", sep="\t", error_bad_lines=False)
        self.test_matched=pd.read_csv(self.path / "test_matched.tsv", sep="\t", error_bad_lines=False)
        self.test_mismatched = pd.read_csv(self.path / "test_mismatched.tsv", sep="\t", error_bad_lines=False)
        self.train = pd.read_csv(self.path / "train.tsv", sep="\t", error_bad_lines=False)


### Text Similarity tasks

class STS_B(TextSimilarity):
    def __init__(self, path):
        super().__init__(path / "STS-B")
        self.load_dev_test_train()

    def load_dev_test_train(self):
        import csv
        self.dev=pd.read_csv(self.path / "dev.tsv", sep="\t", error_bad_lines=False)
        self.train=pd.read_csv(self.path / "train.tsv",sep="\t", error_bad_lines=False)
        self.test=pd.read_csv(self.path/ "test.tsv", sep="\t", error_bad_lines=False, quoting=csv.QUOTE_NONE) #engine='python')


### Relevance Ranking tasks

class QNLI(RelevanceRanking):
    def __init__(self, path):
        super().__init__(path / "QNLI")
        self.load_dev_test_train()