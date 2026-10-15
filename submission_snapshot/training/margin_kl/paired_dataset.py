"""Dataset wrapper and collator that carry the sample index through to the trainer.

The paired objective needs to look up a cached reference logit per sample, which requires the
index to survive collation:
  - PairedDatasetWrapper: wraps an existing dataset and adds "sample_index" to each item
  - PairedDataCollator: extends DataCollatorForSupervisedDataset and gathers the sample
    indices and source image ids into the batch
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Any

import torch

from qwenvl.data.data_processor import DataCollatorForSupervisedDataset


class PairedDatasetWrapper:
    """Wrap a dataset and add a `sample_index` field to every item.

    The wrapped dataset is not modified; this is a forwarding layer.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sample = self.dataset[i]  # type: ignore
        sample["sample_index"] = i
        return sample

    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped dataset."""
        return getattr(self.dataset, name)


@dataclass
class PairedDataCollator(DataCollatorForSupervisedDataset):
    """Gather the sample indices and source image ids into the batch, so the trainer can look up the
    reference logits.
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        batch = super().__call__(instances)

        batch["sample_indices"] = torch.tensor(
            [inst.get("sample_index", -1) for inst in instances], dtype=torch.long
        )

        batch["source_image_ids"] = [str(inst.get("source_image_id", "")) for inst in instances]

        return batch
