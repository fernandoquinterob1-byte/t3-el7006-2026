"""Subject-level fold handling and leakage checks."""

import json
from pathlib import Path


def load_folds(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def get_fold(folds: dict, evaluation: int, fold: int) -> tuple[list[str], list[str]]:
    split = folds[f"eval_{evaluation}"][f"fold_{fold}"]
    validation, test = split["val"], split["test"]
    overlap = set(validation).intersection(test)
    if overlap:
        raise ValueError(f"Validation/test subject leakage in eval {evaluation}, fold {fold}.")
    return validation, test


def validate_folds(folds: dict, available_subjects: set[str]) -> dict:
    report = {"folds": [], "subjects_in_folds": set()}
    for evaluation_name, evaluation in folds.items():
        for fold_name, split in evaluation.items():
            validation, test = set(split["val"]), set(split["test"])
            train = available_subjects - validation - test
            overlap = validation.intersection(test)
            if overlap:
                raise ValueError(f"Subject leakage in {evaluation_name}/{fold_name}.")
            if train.intersection(validation) or train.intersection(test) or validation.intersection(test):
                raise ValueError(f"Derived train/validation/test subject leakage in {evaluation_name}/{fold_name}.")
            report["folds"].append(
                {
                    "evaluation": evaluation_name,
                    "fold": fold_name,
                    "train_subjects": len(train),
                    "validation_subjects": len(validation),
                    "test_subjects": len(test),
                    "train_val_overlap": 0,
                    "train_test_overlap": 0,
                    "val_test_overlap": 0,
                }
            )
            report["subjects_in_folds"].update(validation, test)
    report["subjects_in_folds"] = len(report["subjects_in_folds"])
    report["missing_from_dataset"] = len(
        set().union(*[set(split["val"]) | set(split["test"]) for item in folds.values() for split in item.values()])
        - available_subjects
    )
    report["extra_in_dataset"] = len(
        available_subjects
        - set().union(*[set(split["val"]) | set(split["test"]) for item in folds.values() for split in item.values()])
    )
    if report["missing_from_dataset"] or report["extra_in_dataset"]:
        raise ValueError("Dataset subjects and fold subjects do not match exactly.")
    return report
