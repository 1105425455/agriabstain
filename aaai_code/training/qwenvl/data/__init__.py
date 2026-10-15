import re
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_AAAI_ROOT = _THIS_DIR.parent.parent.parent
_BENCHMARK_SFT = _AAAI_ROOT / "benchmark" / "annotations" / "sft"
_BENCHMARK_ROOT = _AAAI_ROOT / "benchmark"


AGRIABSTAIN_NEW_REASON_3REASON = {
    "annotation_path": str((_BENCHMARK_SFT / "sft_train.json").resolve()),
    "data_path": str(_BENCHMARK_ROOT.resolve()),
}

# agriabstain_new_reason_3reason_paired → aaai_code/benchmark/annotations/sft/sft_train_with_source.json
AGRIABSTAIN_NEW_REASON_3REASON_PAIRED = {
    "annotation_path": str((_BENCHMARK_SFT / "sft_train_with_source.json").resolve()),
    "data_path": str(_BENCHMARK_ROOT.resolve()),
}


data_dict = {
    "agriabstain_new_reason_3reason": AGRIABSTAIN_NEW_REASON_3REASON,
    "agriabstain_new_reason_3reason_paired": AGRIABSTAIN_NEW_REASON_3REASON_PAIRED,
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
    configs = data_list(["agriabstain_new_reason_3reason"])
    for config in configs:
        print(config)
