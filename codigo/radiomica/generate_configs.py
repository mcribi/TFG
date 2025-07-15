import os
import json
import itertools

SEARCH_SPACE_FILE = "config/search_space_ml.json"
OUTPUT_FILE = "config/valid_configurations_ml_clinical3.json"

def main():
    with open(SEARCH_SPACE_FILE, "r") as f:
        full_space = json.load(f)

    print("Loaded search_space_ml.json.")

    # scalers list
    scaler_list = full_space.pop("Scaler", [None])
    print(f"Scalers: {scaler_list}")

    # Discover datasets automatically
    DATASET_ROOT_DIR = "./radiomic_data/cv_metric_clinicos3"
    datasets = sorted([
        os.path.join(DATASET_ROOT_DIR, d)
        for d in os.listdir(DATASET_ROOT_DIR)
        if os.path.isdir(os.path.join(DATASET_ROOT_DIR, d))
    ])
    print(f" Found {len(datasets)} datasets in {DATASET_ROOT_DIR}.")

    # configs
    all_configs = []
    for dataset in datasets:
        for scaler in scaler_list:
            for model_name, model_space in full_space.items():
                print(f"\n Processing model: {model_name} with scaler: {scaler}")

                model_keys = list(model_space.keys())
                model_values = [model_space[k] for k in model_keys]
                model_combos = list(itertools.product(*model_values))
                print(f"   ➜ {len(model_combos)} model hyperparam combos")

                for model_combo in model_combos:
                    model_dict = dict(zip(model_keys, model_combo))
                    config = {
                        **model_dict,
                        "dataset_path": dataset,
                        "classifier": model_name,
                        "scaler": scaler
                    }
                    all_configs.append(config)

    print(f"\n Total configurations generated: {len(all_configs)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_configs, f, indent=2)

    print(f" Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
