from ast import literal_eval
from datetime import datetime
from time import time
from typing import List, Dict
from sklearn.metrics import hamming_loss, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from pydantic import BaseModel

from pymongo import MongoClient, errors
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from utils import MongoHandler, EscoSkillFinder
import math
from assistants import EscoOverallResult, GPTOutputState


def calculate_token_counts(messages: List[Dict[str, str]], tokenizer: PreTrainedTokenizerBase) -> int:
    """
    Berechnet die Tokenanzahl für eine Liste von Chat-Nachrichten im OpenAI-Format.

    Args:
        messages (List[Dict]): Liste von Nachrichten, je mit 'role' und 'content'
        tokenizer (PreTrainedTokenizerBase): Ein HuggingFace-Tokenizer

    Returns:
        int: Anzahl der Tokens im flachen Textprompt
    """
    flat_prompt = ""
    for msg in messages:
        flat_prompt += f"{msg['role']}: {msg['content']}\n"
    tokens = tokenizer.tokenize(flat_prompt)
    return len(tokens)


def prepare_set(raw):
    if isinstance(raw, set):
        return {str(s).lower().strip() for s in raw}
    if isinstance(raw, str):
        raw = raw.split(",")
    try:
        return {str(s).replace("'", "").lower().strip() for s in raw}
    except Exception as e:
        return set()


def calculate_jacquard_similarity(set1, set2):
    set1, set2 = prepare_set(set1), prepare_set(set2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0.0


def calculate_recall(set_result, set_test):
    set_result, set_test = prepare_set(set_result), prepare_set(set_test)
    if not set_test:
        return 0.0
    return len(set_result & set_test) / len(set_test)


def calculate_length_penalty(set_result, set_test, alpha=1.0):
    set_result, set_test = prepare_set(set_result), prepare_set(set_test)
    if not set_test:
        return 0.0
    ratio = len(set_result) / len(set_test) if len(set_test) else 0
    return math.exp(-alpha * abs(ratio - 1))


def calculate_graph_similarity(set_result, set_test, graph_dist_func, beta=1.0):
    set_result, set_test = prepare_set(set_result), prepare_set(set_test)
    if not set_result or not set_test:
        return 0.0
    distances = []
    for r in set_result:
        try:
            min_dist = min(graph_dist_func(r, t) for t in set_test)
            distances.append(min_dist)
        except Exception as e:
            pass
    avg_dist = sum(distances) / len(distances) if distances else float("inf")
    return math.exp(-beta * avg_dist)


def combined_score(set_result, set_test, graph_dist_func, alpha=1.0, beta=1.0):
    recall = calculate_recall(set_result, set_test)
    penalty = calculate_length_penalty(set_result, set_test, alpha)
    sim_score = calculate_graph_similarity(set_result, set_test, graph_dist_func, beta)
    return recall * penalty * sim_score


def serialize_gpt_output_state(state: GPTOutputState) -> dict:
    return {
        "results": [
            s.model_dump() if isinstance(s, BaseModel) else s
            for s in state.get("results", [])
        ]
    }


def calculate_accuracy(found_skills, expected_skills) -> float:
    """
    Berechnet die Genauigkeit der gefundenen Fähigkeiten im Vergleich zu den erwarteten Fähigkeiten.
    :param found_skills: Liste der gefundenen Fähigkeiten
    :param expected_skills: Liste der erwarteten Fähigkeiten
    :return: Genauigkeit als Prozentsatz
    """
    if not expected_skills:
        return 0.0
    correct = sum(1 for s in found_skills if s.lower() in expected_skills)
    return correct / len(expected_skills) * 100.0


def check_esco_skills(skills) -> ([str], [str]):
    """
    Überprüft ob die gefundenen Fähigkeiten in der ESCO-Datenbank vorhanden sind.
    :param found_skills:
    :return: Liste der gefundenen Fähigkeiten, Liste der nicht gefundenen Fähigkeiten
    """
    found_skills = []
    not_found_skills = []

    for skill in skills:
        if isinstance(skill, str):
            skill = skill.strip()
            if EscoSkillFinder().check_label_in_esco_preferredLabel_or_altLabel(skill):
                found_skills.append(skill)
            else:
                not_found_skills.append(skill)
        else:
            not_found_skills.append(skill)

    return found_skills, not_found_skills


def get_num_missing_skills(found_skills, expected_skills) -> int:
    return sum(1 for s in expected_skills if s not in found_skills)


class ExperimentRow:
    input = ""
    expected_output = []
    predicted_output = []
    error = 0
    esco_predictions = []
    not_esco_predictions = []

    def __init__(self, input, expected_output, predicted_output, error=0, not_in_esco_count=0):
        self.input = input
        self.expected_output = self._clean_labels(expected_output)
        self.predicted_output = self._clean_labels(predicted_output)
        self.error = error
        self.not_in_esco_count = not_in_esco_count
        self.in_esco_count = len(self.predicted_output)

    def _clean_labels(self, label_list):
        return [label.strip().replace('\n', '') for label in label_list]

    def _calculate_hamming_loss(self):
        mlb = MultiLabelBinarizer()
        if not isinstance(self.expected_output, list):
            print(f"Received not a list: {self.expected_output}")
            self.expected_output = literal_eval(self.expected_output)
        if not isinstance(self.predicted_output, list):
            print(f"Received not a list: {self.predicted_output}")
            self.predicted_output = literal_eval(self.predicted_output)
        combined = [self.expected_output, self.predicted_output]  # Liste von Listen
        mlb.fit(combined)
        # print(mlb.classes_)
        y_true = mlb.transform([self.expected_output])
        y_pred = mlb.transform([self.predicted_output])
        return hamming_loss(y_true, y_pred)

    def _calculate_accuracy(self):
        accuracy_list = [
            1 if i in self.expected_output else 0 for i in self.predicted_output
        ]
        return sum(accuracy_list) / len(self.expected_output)

    def _calculate_overlap_count(self):
        return len(
            set(self.expected_output).intersection(set(self.predicted_output))
        )

    def _calculate_count_difference_penalty(self):
        total_possible_labels = 123
        return abs(len(self.expected_output) - len(self.predicted_output)) / total_possible_labels

    def _calculate_count_difference_penalty_local(self):
        return abs(len(self.expected_output) - len(self.predicted_output))

    def _calculate_classification_report(self):
        mlb = MultiLabelBinarizer()
        combined = [self.expected_output, self.predicted_output]  # Liste von Listen
        mlb.fit(combined)
        y_true = mlb.transform([self.expected_output])
        y_pred = mlb.transform([self.predicted_output])
        return classification_report(y_true, y_pred, zero_division=0)

    def _calculate_multilabel_confusion_matrix(self):
        mlb = MultiLabelBinarizer()
        combined = [self.expected_output, self.predicted_output]  # Liste von Listen
        mlb.fit(combined)
        y_true = mlb.transform([self.expected_output])
        y_pred = mlb.transform([self.predicted_output])
        return multilabel_confusion_matrix(y_true, y_pred)

    def _calculate_recall(self):
        mlb = MultiLabelBinarizer()
        combined = [self.expected_output, self.predicted_output]  # Liste von Listen
        mlb.fit(combined)
        y_true = mlb.transform([self.expected_output])
        y_pred = mlb.transform([self.predicted_output])
        return (classification_report(y_true, y_pred, zero_division=0, output_dict=True)["macro avg"]["recall"],
                classification_report(y_true, y_pred, zero_division=0, output_dict=True)["micro avg"]["recall"],
                classification_report(y_true, y_pred, zero_division=0, output_dict=True)["weighted avg"]["recall"])

    def _calculate_precision(self):
        mlb = MultiLabelBinarizer()
        combined = [self.expected_output, self.predicted_output]  # Liste von Listen
        mlb.fit(combined)
        y_true = mlb.transform([self.expected_output])
        y_pred = mlb.transform([self.predicted_output])
        return (classification_report(y_true, y_pred, zero_division=0, output_dict=True)["macro avg"]["precision"],
                classification_report(y_true, y_pred, zero_division=0, output_dict=True)["micro avg"]["precision"],
                classification_report(y_true, y_pred, zero_division=0, output_dict=True)["weighted avg"]["precision"])

    def _calculate_f1(self):
        mlb = MultiLabelBinarizer()
        combined = [self.expected_output, self.predicted_output]  # Liste von Listen
        mlb.fit(combined)
        y_true = mlb.transform([self.expected_output])
        y_pred = mlb.transform([self.predicted_output])
        return (classification_report(y_true, y_pred, zero_division=0, output_dict=True)["macro avg"]["f1-score"],
                classification_report(y_true, y_pred, zero_division=0, output_dict=True)["micro avg"]["f1-score"],
                classification_report(y_true, y_pred, zero_division=0, output_dict=True)["weighted avg"]["f1-score"])

    def calculate_metrics(self):
        self.accuracy = self._calculate_accuracy()
        self.overlap_count = self._calculate_overlap_count()
        self.num_predicted = len(self.predicted_output)
        self.count_difference_penalty_global = self._calculate_count_difference_penalty()
        self.count_difference_penalty_local = self._calculate_count_difference_penalty_local()
        self.recall_macro, self.recall_micro, self.recall_weighted = self._calculate_recall()
        self.precision_macro, self.precision_micro, self.precision_weighted = self._calculate_precision()
        self.f1_macro, self.f1_micro, self.f1_weighted = self._calculate_f1()
        self.hamming_loss = self._calculate_hamming_loss()
        self.classification_report = self._calculate_classification_report()
        self.multilabel_confusion_matrix = self._calculate_multilabel_confusion_matrix()

    def to_dict(self):
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "predicted_output": self.predicted_output,
            "error": self.error,
            "esco_predictions": self.esco_predictions,
            "not_esco_predictions": self.not_esco_predictions,
            "not_in_esco_count": getattr(self, "not_in_esco_count", 0),
            "in_esco_count": getattr(self, "in_esco_count", -1),
            "accuracy": getattr(self, "accuracy", None),
            "overlap_count": getattr(self, "overlap_count", None),
            "num_predicted": getattr(self, "num_predicted", None),
            "count_difference_penalty_global": getattr(self, "count_difference_penalty_global", None),
            "count_difference_penalty_local": getattr(self, "count_difference_penalty_local", None),
            "recall_macro": getattr(self, "recall_macro", None),
            "recall_micro": getattr(self, "recall_micro", None),
            "recall_weighted": getattr(self, "recall_weighted", None),
            "precision_macro": getattr(self, "precision_macro", None),
            "precision_micro": getattr(self, "precision_micro", None),
            "precision_weighted": getattr(self, "precision_weighted", None),
            "f1_macro": getattr(self, "f1_macro", None),
            "f1_micro": getattr(self, "f1_micro", None),
            "f1_weighted": getattr(self, "f1_weighted", None),
            "hamming_loss": getattr(self, "hamming_loss", None),
            "classification_report": getattr(self, "classification_report", None),
            "multilabel_confusion_matrix": self.multilabel_confusion_matrix.tolist() if hasattr(self,
                                                                                                "multilabel_confusion_matrix") else None,
        }


class ExperimentResult:
    collection_name = ""
    model = ""
    assistant_str = ""
    few_shot = 0
    experiment_rows = []

    def __init__(self, model, assistant_str, few_shot, collection_name):
        self.collection_name = collection_name
        self.model = model
        self.assistant_str = assistant_str
        self.few_shot = few_shot
        self.experiment_rows = []

    def add_experiment_row(self, experiment_row):
        self.experiment_rows.append(experiment_row)

    def set_experiment_rows(self, experiment_rows):
        self.experiment_rows = experiment_rows

    def calculate_mean_metrics(self):
        for row in self.experiment_rows:
            row.calculate_metrics()
        self.mean_accuracy = sum([row.accuracy for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_overlap_count = sum([row.overlap_count for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_num_predicted = sum([row.num_predicted for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_count_difference_penalty_global = sum(
            [row.count_difference_penalty_global for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_count_difference_penalty_local = sum(
            [row.count_difference_penalty_local for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_hamming_loss = sum([row.hamming_loss for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_not_in_esco_count = sum([row.not_in_esco_count for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_in_esco_count = sum([row.in_esco_count for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_recall_macro = sum([row.recall_macro for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_recall_micro = sum([row.recall_micro for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_recall_weighted = sum([row.recall_weighted for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_precision_macro = sum([row.precision_macro for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_precision_micro = sum([row.precision_micro for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_precision_weighted = sum([row.precision_weighted for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_f1_macro = sum([row.f1_macro for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_f1_micro = sum([row.f1_micro for row in self.experiment_rows]) / len(self.experiment_rows)
        self.mean_f1_weighted = sum([row.f1_weighted for row in self.experiment_rows]) / len(self.experiment_rows)
        self.error_count = sum([row.error for row in self.experiment_rows])
        # self.print_mean_metrics()

    def print_mean_metrics(self):
        print("_" * 10 + self.collection_name + "_" * 10)
        print(f"Mean Accuracy: {self.mean_accuracy * 100:.4f}%")
        print("-" * (20 + len(self.collection_name)))
        print(f"Mean Recall Macro: {self.mean_recall_macro * 100:.4f}%")
        print(f"Mean Recall Micro: {self.mean_recall_micro * 100:.4f}%")
        print(f"Mean Recall Weighted: {self.mean_recall_weighted * 100:.4f}%")
        print("-" * (20 + len(self.collection_name)))
        print(f"Mean Precision Macro: {self.mean_precision_macro * 100:.4f}%")
        print(f"Mean Precision Micro: {self.mean_precision_micro * 100:.4f}%")
        print(f"Mean Precision Weighted: {self.mean_precision_weighted * 100:.4f}%")
        print("-" * (20 + len(self.collection_name)))
        print(f"Mean F1 Macro: {self.mean_f1_macro * 100:.4f}%")
        print(f"Mean F1 Micro: {self.mean_f1_micro * 100:.4f}%")
        print(f"Mean F1 Weighted: {self.mean_f1_weighted * 100:.4f}%")
        print("-" * (20 + len(self.collection_name)))
        print(f"Mean Hamming Loss: {self.mean_hamming_loss}")
        print("-" * (20 + len(self.collection_name)))
        print(f"Mean Number of Predicted Skills: {self.mean_num_predicted}")
        print(f"Mean Not in ESCO Count: {self.mean_not_in_esco_count}")
        print(f"Mean Overlap Count: {self.mean_overlap_count}")
        print("-" * (20 + len(self.collection_name)))
        print(f"Mean Count Difference Penalty Local: {self.mean_count_difference_penalty_local}")
        print(f"Mean Count Difference Penalty Global: {self.mean_count_difference_penalty_global}")
        print("-" * (20 + len(self.collection_name)))
        print(f"Error Count: {self.error_count}")

    def to_dict(self):
        return {
            "collection_name": self.collection_name,
            "model": self.model,
            "assistant_str": self.assistant_str,
            "few_shot": self.few_shot,
            "experiment_rows": [row.to_dict() for row in self.experiment_rows],
            "mean_accuracy": getattr(self, "mean_accuracy", None),
            "mean_overlap_count": getattr(self, "mean_overlap_count", None),
            "mean_num_predicted": getattr(self, "mean_num_predicted", None),
            "mean_count_difference_penalty_global": getattr(self, "mean_count_difference_penalty_global", None),
            "mean_count_difference_penalty_local": getattr(self, "mean_count_difference_penalty_local", None),
            "mean_hamming_loss": getattr(self, "mean_hamming_loss", None),
            "mean_not_in_esco_count": getattr(self, "mean_not_in_esco_count", None),
            "mean_in_esco_count": getattr(self, "mean_in_esco_count", None),
            "mean_recall_macro": getattr(self, "mean_recall_macro", None),
            "mean_recall_micro": getattr(self, "mean_recall_micro", None),
            "mean_recall_weighted": getattr(self, "mean_recall_weighted", None),
            "mean_precision_macro": getattr(self, "mean_precision_macro", None),
            "mean_precision_micro": getattr(self, "mean_precision_micro", None),
            "mean_precision_weighted": getattr(self, "mean_precision_weighted", None),
            "mean_f1_macro": getattr(self, "mean_f1_macro", None),
            "mean_f1_micro": getattr(self, "mean_f1_micro", None),
            "mean_f1_weighted": getattr(self, "mean_f1_weighted", None),
            "error_count": getattr(self, "error_count", None)
        }


class Experiment:
    def __init__(
            self,
            assistant,
            input_data,
            evaluation_data,
            name,
            mongo_uri="mongodb://localhost:27017",
            db_name="D_ExperimentResults" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            collection_name=None,
    ):
        """
        Wenn collection_name nicht übergeben wird, wird sie aus name abgeleitet.
        """
        if len(input_data) != len(evaluation_data):
            raise ValueError("Input data and evaluation data must have the same length.")
        if not input_data or not evaluation_data:
            raise ValueError("Input data and evaluation data must not be empty.")

        self.assistant = assistant
        self.input_data = input_data
        self.evaluation_data = evaluation_data
        self.name = name
        self.timestamp = time()

        # MongoDB-Setup
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        # automatisch abgeleiteter Collection-Name, falls nicht gesetzt
        self.collection_name = collection_name or f"{name}_{int(self.timestamp)}"
        # nur anlegen, wenn nicht existent
        if self.collection_name not in self.db.list_collection_names():
            try:
                self.db.create_collection(self.collection_name)
            except Exception as e:
                tqdm.write("Error creating collection" + self.collection_name)

        self.collection = self.db[self.collection_name]

    def update_prompt(self, prompt: str):
        self.assistant.overwrite_prompt_and_build_graph(prompt)

    def run_experiment(self):
        experiment_start_time = time()
        self.result = ExperimentResult(model=self.assistant.model_name,
                                       assistant_str=self.name.split("_")[0],
                                       few_shot="fewshot" in self.name.lower(),
                                       collection_name=self.collection_name, )

        for X, y in tqdm(zip(self.input_data, self.evaluation_data), total=len(self.input_data), desc="Processing",
                         position=4, leave=True):
            run_start_time = time()
            skills: GPTOutputState = self.assistant.invoke({"description": X})

            run_end_time = time()
            total_run_time = run_end_time - run_start_time

            # get preferredLabels from skills
            predicted_labels = []
            try:
                for skillResult in skills.get("results", []):
                    if isinstance(skillResult, dict):
                        skill = skillResult["preferredLabel"]
                        predicted_labels.append(skill)
            except Exception as e:
                print(f"Experiment Parsing Error: {e}")
                predicted_labels.append(skills)

            # check if predicted in esco
            esco_predictions = []
            not_esco_predictions = []
            for label_ in predicted_labels:
                if isinstance(label_, str):
                    esco_predictions.append(label_) if EscoSkillFinder().check_label_in_esco_preferredLabel(
                        label_) else not_esco_predictions.append(label_)
                else:
                    not_esco_predictions.append(str(label_))

            # add raw result to row
            current = ExperimentRow(input=X,
                                    expected_output=y,
                                    predicted_output=esco_predictions,
                                    not_in_esco_count=len(set(not_esco_predictions)))
            current.esco_predictions = esco_predictions
            current.not_esco_predictions = not_esco_predictions
            current.calculate_metrics()

            # add current to result
            self.result.add_experiment_row(current)

            # store current row in MongoDB document
            current_row_document = current.to_dict()
            current_row_document["timestamp"] = self.timestamp
            current_row_document["experiment_name"] = self.name
            current_row_document["input"] = X
            current_row_document["expected_output"] = y
            current_row_document["run_time_s"] = total_run_time
            try:
                current_row_document["raw_output"] = serialize_gpt_output_state(skills)
            except Exception as e:
                print(f"Error serializing GPT output state: {e}")
                current_row_document["raw_output"] = str(skills)

            try:
                self.collection.insert_one(current_row_document)
            except Exception as e:
                print(f"Fehler beim Einfügen in MongoDB: {e}")

        self.summarize_experiment_results()


    def summarize_experiment_results(self):
        self.result.calculate_mean_metrics()
        current_document = self.result.to_dict()

        try:
            self.collection.insert_one(current_document)
        except Exception as e:
            tqdm.write("Fehler beim Einfügen der Zusammenfassung in MongoDB: " + str(e))

