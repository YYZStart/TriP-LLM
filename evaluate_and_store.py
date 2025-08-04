import os
from ray.rllib.core.models.specs.specs_base import torch
from pate.PATE_metric import PATE
from preprocessing import data_processing


def evaluate_and_store(test_score, truth_labels,
                       file_key="Dataset_subset_name",
                       model_name="model"):
    # Since we already have the anomaly scores for the testing set,
    # point-wise F1 score, composite F1 score, and Affiliation-based F1 scores
    # can be easily computed using a given thresholding function.
    #
    # Here, we only compute the PATE score.
    # You may customize this section to include other evaluation metrics or thresholding methods as needed.

    if truth_labels.dtype == torch.bfloat16:
        truth_labels = truth_labels.to(torch.float32)

    truth_labels = truth_labels.cpu().numpy()

    pate = PATE(truth_labels, test_score, binary_scores=False)

    print(f"PATE: {round(pate, 4)}")

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    file_name = os.path.join(model_name, f"{model_name}_{file_key}.txt")

    with open(file_name, 'w') as file:

        file.write(f"\nPATE: {round(pate, 4)}\n")
        file.write(f"{model_name}_{file_key}\n")


def generate_data_for_ad(Dataset_name, test_set_name, data_type=torch.float32):
    X_train, y_train, X_test, y_test = data_processing(Dataset_name, test_set_name, data_type)

    return X_train, X_test, y_test


def model_inferences_ad(testset, truth_label, file_key, clf, model_name):
    scores = clf.decision_function(testset)

    evaluate_and_store(scores, truth_label, file_key=file_key, model_name=model_name)

    print(f"{model_name}_{file_key} is completed!!")
