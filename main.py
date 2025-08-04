import os
import gc
import torch
from evaluate_and_store import generate_data_for_ad, model_inferences_ad
from model.TriP_LLM import TriP_LLM

dim_datasets = { #dataset dimension
    "SMD": 38,
    "SWaT": 51,
    "MSL": 55,
    "SMAP": 25,
    "PSM": 25,
    "NIPS_TS_SWAN": 38,
    "NIPS_TS_GECCO": 9,
}

test_set_name = "MN" # Change according to your dataset

data_type = torch.float32

model_path = "gpt2" # Change if you want other LLMs, check  Util.py

ablations = [
    "",
]

use_amp = data_type != torch.bfloat16  # True → use torch.cuda.amp

hyper_parm = {

    "PSM": dict(batch_size=256, lr=0.0004, seq_len=96, epochs=10,
                patch_size=[8, 32], patch_stride=3),

    "SMD": dict(batch_size=256, lr=0.0001, seq_len=64, epochs=5,
                patch_size=[8, 16, 32], patch_stride=3),

    "SWaT": dict(batch_size=128, lr=0.0002, seq_len=48, epochs=5,
                 patch_size=[4, 8, 12, 16], patch_stride=2),

    "MSL": dict(batch_size=64, lr=0.0001, seq_len=48, epochs=5,
                patch_size=[4, 8, 12], patch_stride=2),

    "NIPS_TS_SWAN": dict(batch_size=64, lr=2e-4, seq_len=64, epochs=5,
                         patch_size=[4, 8, 16, 32], patch_stride=2),
}

# shared by all configs
common = dict(
    pred_len=32,  # unused params
    data_type=data_type,
    finetune=False,
    use_amp=use_amp,
    model_path=model_path,
)

dataset_params = {}
for name, feat_dim in dim_datasets.items():
    # Use default settings; override if other settings exist
    cfg = dict(
        batch_size=64,
        lr=1e-4,
        seq_len=96,
        epochs=10,
        patch_size=[8, 16],
        patch_stride=2,
        feature_dim=feat_dim,
        **common
    )
    if name in hyper_parm:  # override
        cfg.update(hyper_parm[name])
        print(f"Loaded settings for dataset: {name}")

    dataset_params[name] = cfg

if __name__ == '__main__':
    selected_model = TriP_LLM
    selected_model_name = "TriP_LLM"

    datasets = [
        "MSL",
        "SMD",
        "SWaT",
        "PSM",
        "NIPS_TS_SWAN",
    ]

    action = "infer_only"
    # "train_and_infer" or "infer_only"

    for ablation in ablations:
        print(f"\n=====  Start ablation: {ablation}  =====")

        model_configs = {
            selected_model_name: {
                "model": selected_model,
                "params": {"device": "cuda"},
            }
        }

        for Dataset_name in datasets:

            dataset_specific_params = dataset_params[Dataset_name].copy()
            dataset_specific_params["ablation"] = ablation

            model_configs[selected_model_name]["params"].update(dataset_specific_params)

            print(f"Dataset: {Dataset_name}, Config: {model_configs[selected_model_name]['params']}")

            # generate preprocessed data
            X_train, MN_test, MN_label = generate_data_for_ad(
                Dataset_name, test_set_name, data_type=data_type
            )

            for model_name, config in model_configs.items():
                model_dir = os.path.join(r"Checkpoints", model_name)
                os.makedirs(model_dir, exist_ok=True)

                prefix_str = ("Finetuned_" if dataset_specific_params["finetune"] else "") # if fine-tuned the model
                prefix_str = f"{ablation}{prefix_str}"

                model_file = os.path.join(
                    model_dir,
                    f"{prefix_str}{model_path}_{model_name}_{Dataset_name}.pt"
                )

                if action == "infer_only" and os.path.exists(model_file):
                    print(f"Loading model for inference: {model_file}")
                    clf = config["model"](**config["params"])
                    clf.load_pt_model(model_file)

                elif action == "train_and_infer":
                    print(f"Training model for dataset {Dataset_name} ...")
                    clf = config["model"](**config["params"])
                    clf.fit(X_train)
                    print(f"Saving trained model to {model_file}")
                    clf.save_pt_model(model_file)

                    param_file = os.path.join(model_dir, "dataset_params_log.txt")
                    with open(param_file, "a", encoding="utf-8") as f:
                        f.write(f"=== Parameters for {Dataset_name} | ablation={ablation} ===\n")
                        for k, v in dataset_specific_params.items():
                            f.write(f"{k}: {v}\n")
                        f.write("\n")

                else:
                    print(f"Model file not found for inference: {model_file}")
                    continue

                # inference
                model_inferences_ad(
                    MN_test, MN_label,
                    file_key=f"{Dataset_name}_{test_set_name}",
                    clf=clf,
                    model_name=f"{prefix_str}{model_path}_{model_name}",
                )

                del clf
                gc.collect()
                torch.cuda.empty_cache()

            print(f"=== Finished dataset: {Dataset_name} | ablation: {ablation} ===\n")

        print(f"=====  End   ablation: {ablation}  =====\n")
