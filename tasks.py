import datasets

from mteb.abstasks import AbsTaskBitextMining
from mteb.abstasks.AbsTaskBitextMining import BitextMiningEvaluator
from mteb.abstasks.AbsTaskClassification import AbsTaskClassification
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval


class BornholmBitextMining(AbsTaskBitextMining):
    @property
    def description(self):
        return {
            "name": "Bornholm Parallel Corpus",
            "hf_hub_name": "strombergnlp/bornholmsk_parallel",
            "description": "Bornholm Parallel Corpus",
            "reference": "https://aclanthology.org/W19-6138/",
            "type": "BitextMining",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "main_score": "f1",
            "revision": "3bc5cfb4ec514264fe2db5615fac9016f7251552",
        }

    def _evaluate_split(self, model, data_split, **kwargs):
        sent_1 = "da_bornholm"
        sent_2 = "da"
        if len(data_split[sent_1]) == 1:
            sentence1 = data_split[sent_1][0]
        else:
            sentence1 = data_split[sent_1]
        if len(data_split[sent_2]) == 1:
            sentence2 = data_split[sent_2][0]
        else:
            sentence2 = data_split[sent_2]

        if not ("gold" in data_split.features):
            assert len(sentence1) == len(sentence2), "Wrong dataset format"
            n = len(sentence1)
            gold = list(zip(range(n), range(n)))
        else:
            gold = data_split["gold"]
            if len(gold) == 1:
                gold = gold[0]
            # MTEB currently only loads GOLD labels for BUCC, which is 1-indexed
            # If a 2nd 0-indexed dataset is added, it'd be cleaner to update BUCC on the Hub to be 0-indexed
            gold = [(i - 1, j - 1) for (i, j) in gold]
            assert all(
                [(i > 0) and (j > 0) for i, j in gold]
            ), "Found negative gold indices. This may be caused by MTEB expecting 1-indexed gold labels."

        evaluator = BitextMiningEvaluator(sentence1, sentence2, gold, **kwargs)
        metrics = evaluator(model)
        self._add_main_score(metrics)
        return metrics

    def _add_main_score(self, scores):
        if self.description["main_score"] in scores:
            scores["main_score"] = scores[self.description["main_score"]]
        else:
            print(f"WARNING: main score {self.description['main_score']} not found in scores {scores.keys()}")


class AngryTweetsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "AngryTweetsClassification",
            "hf_hub_name": "DDSC/angry-tweets",
            "description": ("A sentiment dataset with 3 classes (positiv, negativ, neutral) for Danish tweets"),
            "reference": "https://aclanthology.org/2021.nodalida-main.53/",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "type": "Classification",
            "category": "s2s",
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "20b0e6081892e78179356fada741b7afa381443d",
        }


class DKHateClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "DKHateClassification",
            "hf_hub_name": "DDSC/dkhate",
            "description": "Danish Tweets annotated for Hate Speech either being Offensive or not",
            "reference": "https://aclanthology.org/2020.lrec-1.430/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test", "validation"],
            "eval_langs": ["da"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "59d12749a3c91a186063c7d729ec392fda94681c",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])

        self.data_loaded = True


class MainlandScandinavianLangClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "NordicLangClassification",
            "hf_hub_name": "strombergnlp/nordic_langid",
            "description": "A dataset for Nordic language identification",
            "reference": "https://aclanthology.org/2021.vardial-1.8/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da", "no", "sv", "nb", "no"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "e254179d18ab0165fdb6dbef91178266222bee2a",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], "10k", revision=self.description.get("revision")
        )
        self.data_loaded = True
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("language", "label")

        # only use languages "da", "no", "sv", "nb", "no"
        langs_to_keep = {"da", "no", "sv", "nb", "no"}
        lang_feature = self.dataset["test"].info.features["label"].names  # type: ignore
        idx2lang = {idx: lang for idx, lang in enumerate(lang_feature) if lang in langs_to_keep}
        self.dataset = self.dataset.filter(lambda x: x["label"] in idx2lang)


class DanishPoliticalCommentsClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "DanishPoliticalCommentsClassification",
            "hf_hub_name": "danish_political_comments",
            "description": "A dataset of Danish political comments rated for sentiment",
            "reference": "NA",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["train"],
            "eval_langs": ["da"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "edbb03726c04a0efab14fc8c3b8b79e4d420e5a1",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision")
        )
        self.data_loaded = True
        self.dataset = self.dataset.rename_column("sentence", "text")
        self.dataset = self.dataset.rename_column("target", "label")


class LccClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "LccClassification",
            "hf_hub_name": "DDSC/lcc",
            "description": "The leipzig corpora collection, annotated for sentiment",
            "reference": "https://github.com/fnielsen/lcc-sentiment",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "de7ba3406ee55ea2cc52a0a41408fa6aede6d3c6",
        }


class NorwegianParliamentClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "NorwegianParliament",
            "hf_hub_name": "NbAiLab/norwegian_parliament",
            "description": "Norwegian parliament speeches annotated for sentiment",
            "reference": "https://huggingface.co/datasets/NbAiLab/norwegian_parliament",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test", "validation"],
            "eval_langs": ["no"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "f7393532774c66312378d30b197610b43d751972",
        }


class ScalaDaClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ScalaDaClassification",
            "hf_hub_name": "ScandEval/scala-da",
            "description": "A modified version of DDT modified for linguistic acceptability classification",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["da"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "1de08520a7b361e92ffa2a2201ebd41942c54675",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])

        self.data_loaded = True


class ScalaNbClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ScalaNbClassification",
            "hf_hub_name": "ScandEval/scala-nb",
            "description": "A Norwegian dataset for linguistic acceptability classification for Bokmål",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["no", "nb"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "237111a078ad5a834a55c57803d40bbe410ed03b",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])

        self.data_loaded = True


class ScalaNnClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ScalaNbClassification",
            "hf_hub_name": "ScandEval/scala-nn",
            "description": "A Norwegian dataset for linguistic acceptability classification for Nynorsk",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["no", "nn"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "9d9a2a4092ed3cacf0744592f6d2f32ab8ef4c0b",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])

        self.data_loaded = True


class ScalaSvClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "ScalaSvClassification",
            "hf_hub_name": "ScandEval/scala-sv",
            "description": "A Swedish dataset for linguistic acceptability classification",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "1b48e3dcb02872335ff985ff938a054a4ed99008",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], revision=self.description.get("revision", None)
        )
        # convert label to a 0/1 label
        labels = self.dataset["train"]["label"]  # type: ignore
        lab2idx = {lab: idx for idx, lab in enumerate(set(labels))}
        self.dataset = self.dataset.map(lambda x: {"label": lab2idx[x["label"]]}, remove_columns=["label"])

        self.data_loaded = True


class SweRecClassificition(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "SwerecClassification",
            "hf_hub_name": "ScandEval/swerec-mini",
            "description": "A Swedish dataset for sentiment classification on review",
            "reference": "https://aclanthology.org/2023.nodalida-1.20/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "3c62f26bafdc4c4e1c16401ad4b32f0a94b46612",
        }


class NoRecClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "NoRecClassification",
            "hf_hub_name": "ScandEval/norec-mini",
            "description": "A Norwegian dataset for sentiment classification on review",
            "reference": "https://aclanthology.org/L18-1661/",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["no"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "07b99ab3363c2e7f8f87015b01c21f4d9b917ce3",
        }


# SuperLIM tasks
class DalajClassification(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "DalajClassification",
            "hf_hub_name": "AI-Sweden/SuperLim",
            "description": "A Swedish dataset for linguistic accebtablity. Available as a part of Superlim",
            "reference": "https://spraakbanken.gu.se/en/resources/superlim",
            "type": "Classification",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 16,
            "revision": "7ebf0b4caa7b2ae39698a889de782c09e6f5ee56",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], "dalaj", revision=self.description.get("revision")
        )
        self.data_loaded = True
        self.dataset = self.convert_to_classification_task(self.dataset)

    @staticmethod
    def convert_to_classification_task(dataset):
        """
        This dataset consist of two columns of relevance, "original_sentence" and "corrected_sentence".
        We will use the original sentence as we "wrong" sentence and the corrected sentence as the "correct" sentence
        """

        def __convert_sample_to_classification(sample):
            text = sample["original_sentence"] + sample["corrected_sentence"]
            label = [1] * len(sample["original_sentence"]) + [0] * len(sample["corrected_sentence"])
            return {"text": text, "label": label}

        columns_to_keep = ["original_sentence", "corrected_sentence"]
        for split in dataset:
            columns_names = dataset[split].column_names
            columns_to_remove = [col for col in columns_names if col not in columns_to_keep]
            dataset[split] = dataset[split].remove_columns(columns_to_remove)

        dataset = dataset.map(__convert_sample_to_classification, batched=True, remove_columns=columns_to_keep)
        return dataset


class SweFAQRetrieval(AbsTaskRetrieval):
    """
    AbsTaskRetrieval: Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = Dict[id, Dict[str, str]] #id => dict with document datas like title and text
    self.queries = Dict[id, str] #id => query
    self.relevant_docs = List[id, id, score]
    """

    @property
    def description(self):
        return {
            "name": "SweFAQ",
            "hf_hub_name": "AI-Sweden/SuperLim",
            "description": "A Swedish FAQ dataset. Available as a part of Superlim",
            "reference": "https://spraakbanken.gu.se/en/resources/superlim",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["sv"],
            "main_score": "ndcg_at_10",
            "revision": "7ebf0b4caa7b2ae39698a889de782c09e6f5ee56",
        }

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(
            self.description["hf_hub_name"], "swefaq", revision=self.description.get("revision")
        )
        self.data_loaded = True

        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        for split in self.description["eval_splits"]:
            dataset = self.dataset[split]  # type: ignore
            answers = dataset["candidate_answer"]
            self.corpus[split] = {idx: {"title": "", "text": answer} for idx, answer in enumerate(answers)}

            questions = dataset["question"]
            self.queries[split] = {idx: question for idx, question in enumerate(questions)}
            self.relevant_docs[split] = []  # Naïve approach, no relevant docs


# from datasets import load_dataset

# dataset = load_dataset("AI-Sweden/SuperLim", "swefaq", split="test")
# len(dataset)
# len(dataset["correct_answer"])
# len(set(dataset["correct_answer"]))

# https://huggingface.co/datasets/sbx/superlim-2/viewer/swepar/test
