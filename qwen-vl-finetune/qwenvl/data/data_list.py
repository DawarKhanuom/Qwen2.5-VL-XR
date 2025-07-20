import re
from .__init__ import data_dict  # ✅ Proper import

def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0

def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict:
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"[ERROR] Dataset '{dataset_name}' not found in data_dict")
    return config_list

# Optional test
if __name__ == "__main__":
    configs = data_list(["my_dataset"])
    for config in configs:
        print(config)