"""
!pip install "mteb[beir]"
!pip install git+https://github.com/UKPLab/sentence-transformers.git  --upgrade
!pip install --force --no-deps git+https://github.com/UKPLab/sentence-transformers.git
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from mteb import MTEB
from tasks import (
    AngryTweetsClassification,
    BornholmBitextMining,
    DalajClassification,
    DanishPoliticalCommentsClassification,
    DKHateClassification,
    LccClassification,
    MainlandScandinavianLangClassification,
    NoRecClassification,
    NorwegianParliamentClassification,
    ScalaDaClassification,
    ScalaNbClassification,
    ScalaSvClassification,
    SweFAQRetrieval,
    SweRecClassificition,
)

# dataset selection:
# Wherever possible we try to keep the as streamlined with ScandEval as possible
# if two versions of dataset exist, we try to use the same one
# To ensure validity we do not include machine translated datasets (this exludes the QA datasets in ScandEval)
# As opposed to ScandEval where inclusion of additional datasets come at a high additional cost, we do not have this limitation
# We include additional datasets that are not included in ScandEval
# Where possible we use the test split, if not we use the validation split. If there is no splits we use the train split.
# We want the benchmark to be reproducible by anyone and therefore does not include datasets such as DaNewsroom
# that is not publicly available

# Implementation:
# We build our benchmark using MTEB library to run the benchmark.
# This allow user to run in along with the other benchmarks in MTEB


# Model selection:
# We select a set of representative encoder models from ScandEval
# We additionally select at set of competitive models from the MTEB leaderboard
# We select representative models such as BERT and RoBERTa
# We select a series of representative multilingual models such as mBERT and XLM-R
# from sentence-transformers we select a series a series of baseline models

# Limitations:
# Currently no of the languages have tasks available for
# Bitext (with the exception of Danish)
# Reranking
# Retrieval
# Clustering
# STS (With the exception of Swedish)
# Summarization (Danish has DaNewsroom but it is not publicly available).

#


model_names = [
    "sentence-transformers/all-MiniLM-L6-v2",
    # "KBLab/sentence-bert-swedish-cased",
    # "jzju/sbert-sv-lim2",
    # "chcaa/dfm-encoder-large-v1",
    # "vesteinn/DanskBERT",
    # "KennethEnevoldsen/dfm-sentence-encoder-medium",
    # "KennethEnevoldsen/dfm-sentence-encoder-medium-2",
    # "KennethEnevoldsen/dfm-sentence-encoder-medium-3",
    # "intfloat/e5-base",
    # "intfloat/multilingual-e5-small",
    # "intfloat/multilingual-e5-base",
    # "intfloat/multilingual-e5-large",
    # "KennethEnevoldsen/dfm-sentence-encoder-large-1",
    # "KennethEnevoldsen/dfm-sentence-encoder-large-2",
]


outputs = []
for model_name in model_names:
    output_folder = f"results/{model_name}"
    outputs.append(output_folder)
    model = SentenceTransformer(model_name)

    evaluation = MTEB(task_langs=["da"])
    evaluation.run(model, output_folder=output_folder)
    # custom tasks
    evaluation = MTEB(
        task_langs=["da"],
        tasks=[
            AngryTweetsClassification(),
            DKHateClassification(),
            BornholmBitextMining(),
            DanishPoliticalCommentsClassification(),
            LccClassification(),
            MainlandScandinavianLangClassification(),
            NoRecClassification(),
            NorwegianParliamentClassification(),
            ScalaDaClassification(),
            ScalaNbClassification(),
            ScalaSvClassification(),
            SweFAQRetrieval(),
            SweRecClassificition(),
            DalajClassification(),
        ],
    )
    evaluation.run(model, output_folder=output_folder)


def load_scores_from_path(path):
    # find all json files in path
    json_files = list(Path(path).rglob("*.json"))
    # load json files
    scores = []
    for json_file in json_files:
        with open(json_file) as f:
            scores.append(json.load(f))
    return scores


def extract_main_perf_metric(output):
    scores = []
    for task in output:
        for split in [
            # "train", "dev", "validation",
            "test"
        ]:
            if split not in task:
                continue
            main_score = task[split].get("da", task[split])["main_score"]
            task_name = task["mteb_dataset_name"]

            score = {"task": task_name, "split": split, "main_score": main_score}
            scores.append(score)
    return scores


scores = [load_scores_from_path(output) for output in outputs]
scores = [extract_main_perf_metric(output) for output in scores]

for mdl_nam, score in zip(model_names, scores):
    print(mdl_nam)
    main_scores = [s["main_score"] for s in score]
    print(f"{np.mean(main_scores):.3f}±{np.std(main_scores):.3f}")

    for s in score:
        print(s)
