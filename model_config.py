from pathlib import Path


LABELS = [
    "caution_and_advice",
    "displaced_people_and_evacuations",
    "donation_and_volunteering",
    "infrastructure_and_utilities_damage",
    "injured_or_dead_people",
    "missing_trapped_or_found_people",
    "not_related_or_irrelevant",
    "other_useful_information",
    "sympathy_and_emotional_support",
]

DEFAULT_DATA_DIR = Path("data")
DEFAULT_LEXICON_PATH = DEFAULT_DATA_DIR / "CrisisLexLexicon" / "CrisisLexRec.txt"
DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 128
DEFAULT_BASELINE_OUTPUT_DIR = Path("outputs/bert_cross_entropy")
DEFAULT_WEIGHTED_OUTPUT_DIR = Path("outputs/bert_weighted_cross_entropy")
DEFAULT_LABEL_SMOOTHING_OUTPUT_DIR = Path("outputs/bert_label_smoothing")
DEFAULT_EXTENSION_OUTPUT_DIR = Path("outputs/bert_weighted_label_smoothing")


def build_label_mappings():
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label
