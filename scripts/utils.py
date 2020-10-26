import pandas as pd
import json
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
        self.dev_jsonl=pd.read_json(self.path / "dev.jsonl", orient='records')
        self.test_jsonl = pd.read_json(self.path / "test.jsonl", orient='records')
        self.train_jsonl = pd.read_json(self.path / "train.jsonl", orient="records")


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

class STS_B(TextSimilarity):
    def __init__(self, path):
        super().__init__(path / "STS-B")
        self.load_dev_test_train()

    def load_dev_test_train(self):
        self.dev=pd.read_csv(self.path / "dev.tsv", sep="\t", error_bad_lines=False)
        self.train=pd.read_csv(self.path / "train.tsv",sep="\t", error_bad_lines=False)
        self.test=pd.read_csv(self.path/ "test.tsv", sep="\t", error_bad_lines=False, engine='python')

class QNLI(RelevanceRanking):
    def __init__(self, path):
        super().__init__(path / "QNLI")
        self.load_dev_test_train()