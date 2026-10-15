import re
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _THIS_DIR.parent.parent.parent
_BENCHMARK_SFT = _PACKAGE_ROOT / "benchmark" / "annotations" / "sft"
_BENCHMARK_ROOT = _PACKAGE_ROOT / "benchmark"


REGATE_3REASON = {
    "annotation_path": str((_BENCHMARK_SFT / "sft_train.json").resolve()),
    "data_path": str(_BENCHMARK_ROOT.resolve()),
}

# regate_3reason_paired → benchmark/annotations/sft/sft_train_with_source.json
REGATE_3REASON_PAIRED = {
    "annotation_path": str((_BENCHMARK_SFT / "sft_train_with_source.json").resolve()),
    "data_path": str(_BENCHMARK_ROOT.resolve()),
}


data_dict = {
    "regate_3reason": REGATE_3REASON,
    "regate_3reason_paired": REGATE_3REASON_PAIRED,
}


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
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    configs = data_list(["regate_3reason"])
    for config in configs:
        print(config)
