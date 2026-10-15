#!/usr/bin/env python3
"""Split the built benchmark into train, development and test, which carry hard references, plus
a dose pool for the ambiguous intermediate levels.

Input:
- package layout: benchmark/annotations/raw/{plantdoc,coco}_full.raw.jsonl
- source layout:  outputs/benchmarks/{plantdoc,coco}_full.raw.jsonl

Output:
- package layout: benchmark/annotations/splits/{train,dev,test,dose_pool}.jsonl and
  split_manifest.json
- source layout:  outputs/splits/...
"""

import argparse
import json
import os
import random
from collections import defaultdict

# Project root = aaai_code/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Package mode: when scripts/ is inside aaai_code/ and benchmark/ exists alongside
PACKAGE_MODE = os.path.isdir(os.path.join(PROJECT_ROOT, "benchmark"))
if PACKAGE_MODE:
    INPUT_FILES = [
        os.path.join(PROJECT_ROOT, "benchmark/annotations/raw/plantdoc_full.raw.jsonl"),
        os.path.join(PROJECT_ROOT, "benchmark/annotations/raw/coco_full.raw.jsonl"),
    ]
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark/annotations/splits")
else:
    # source mode: fallback for legacy layout
    INPUT_FILES = [
        os.path.join(SCRIPT_DIR, "../outputs/benchmarks/plantdoc_full.raw.jsonl"),
        os.path.join(SCRIPT_DIR, "../outputs/benchmarks/coco_full.raw.jsonl"),
    ]
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../outputs/splits")


TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
TEST_RATIO = 0.15


def load_samples(filepath):
    """Load all samples from a JSONL file."""
    samples = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def classify_sample(sample):
    """Route a sample: `dose_only` goes to the dose pool, everything else is split by decision into
    answer or abstain.
    """
    metadata = sample.get("metadata", {})
    if metadata.get("dose_only") is True:
        return "fuzzy"
    decision = sample.get("decision", "")
    if decision == "answer":
        return "answer"
    if decision == "abstain":
        return "abstain"
    return "fuzzy"


def extract_crop(sample):
    """Take the crop name used for stratification.

    plantdoc: the first word of `source_target.label`
    coco: the single stratum `coco`
    """
    dataset = sample.get("dataset", "")
    if "plantdoc" in dataset:
        label = sample.get("source_target", {}).get("label", "")
        if label:
            return label.split()[0]
        # fallback: try mismatch_label if exists
        mismatch_label = sample.get("metadata", {}).get("mismatch_label", "")
        if mismatch_label:
            return mismatch_label.split()[0]
        return "unknown"
    else:
        return "coco"


def get_split_by_hash(dataset, source_image_id, crop, crop_image_ids):
    """Assign a split by taking the MD5 hash of `source_image_id` modulo the split sizes.

    Every sample derived from one image lands in the same split. Assignment is stratified by
    crop, hashing the image ids of each crop separately.
    """
    sorted_ids = sorted(crop_image_ids)
    idx = sorted_ids.index(source_image_id)
    total = len(sorted_ids)

    train_end = int(total * TRAIN_RATIO)
    dev_end = int(total * (TRAIN_RATIO + DEV_RATIO))

    if idx < train_end:
        return "train"
    elif idx < dev_end:
        return "dev"
    else:
        return "test"


def balanced_sample(samples, target_answer_count, target_abstain_count, keep_all_t6=True):
    """Balance the sample counts.

    - every T6 record, with reason wrong_object, is kept
    - non-T6 abstentions, with reason no_target, are sampled down to the target count
    - answers are sampled down to the target count
    """
    answer_samples = [s for s in samples if s["_category"] == "answer"]
    abstain_samples = [s for s in samples if s["_category"] == "abstain"]

    t6_samples = [s for s in abstain_samples if s.get("metadata", {}).get("relation") == "T6"]
    non_t6_abstain = [s for s in abstain_samples if s.get("metadata", {}).get("relation") != "T6"]

    total_abstain_target = target_abstain_count

    t6_count = len(t6_samples)

    remaining_abstain_slots = max(0, total_abstain_target - t6_count)

    random.seed(42)
    if len(non_t6_abstain) <= remaining_abstain_slots:
        selected_non_t6_abstain = non_t6_abstain
    else:
        selected_non_t6_abstain = random.sample(non_t6_abstain, remaining_abstain_slots)

    selected_abstain = t6_samples + selected_non_t6_abstain

    if len(answer_samples) <= target_answer_count:
        selected_answer = answer_samples
    else:
        base_samples = [
            s for s in answer_samples if s.get("metadata", {}).get("relation") == "base"
        ]
        ladder_100 = [
            s
            for s in answer_samples
            if s.get("metadata", {}).get("relation") == "T7_ladder"
            and s.get("metadata", {}).get("achieved_visible_ratio", 0) >= 0.999
        ]
        other_answer = [s for s in answer_samples if s not in base_samples and s not in ladder_100]

        selected_answer = base_samples + ladder_100
        remaining_answer_slots = max(0, target_answer_count - len(selected_answer))

        if remaining_answer_slots > 0 and other_answer:
            selected_answer += random.sample(
                other_answer, min(remaining_answer_slots, len(other_answer))
            )
        else:
            selected_answer = selected_answer[:target_answer_count]

    return selected_answer, selected_abstain


def main():
    global INPUT_FILES, OUTPUT_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plantdoc",
        default=INPUT_FILES[0],
    )
    parser.add_argument(
        "--coco",
        default=INPUT_FILES[1],
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
    )
    args = parser.parse_args()
    INPUT_FILES = [args.plantdoc, args.coco]
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading samples...")
    all_samples = []
    for filepath in INPUT_FILES:
        samples = load_samples(filepath)
        print(f"  {filepath}: {len(samples)} samples")
        all_samples.extend(samples)

    print(f"Total samples: {len(all_samples)}")

    print("\nClassifying samples...")
    answer_samples = []
    abstain_samples = []
    fuzzy_samples = []

    for s in all_samples:
        cat = classify_sample(s)
        s["_category"] = cat
        if cat == "answer":
            answer_samples.append(s)
        elif cat == "abstain":
            abstain_samples.append(s)
        else:
            fuzzy_samples.append(s)

    print(f"  Answer: {len(answer_samples)}")
    print(f"  Abstain: {len(abstain_samples)}")
    print(f"  Fuzzy: {len(fuzzy_samples)}")

    print("\nWriting dose_pool.jsonl...")
    dose_pool_path = os.path.join(OUTPUT_DIR, "dose_pool.jsonl")
    with open(dose_pool_path, "w") as f:
        for s in fuzzy_samples:
            s_out = {k: v for k, v in s.items() if k != "_category"}
            f.write(json.dumps(s_out, ensure_ascii=False) + "\n")
    print(f"  dose_pool.jsonl: {len(fuzzy_samples)} samples")

    print("\nSplitting answer/abstain samples by source_image_id...")

    crop_image_ids = defaultdict(lambda: defaultdict(set))
    for s in answer_samples + abstain_samples:
        dataset = s.get("dataset", "")
        source_id = s["source_image_id"]
        crop = extract_crop(s)
        crop_image_ids[dataset][crop].add(source_id)

    train_samples = []
    dev_samples = []
    test_samples = []

    for s in answer_samples + abstain_samples:
        dataset = s.get("dataset", "")
        source_id = s["source_image_id"]
        crop = extract_crop(s)

        split = get_split_by_hash(dataset, source_id, crop, crop_image_ids[dataset][crop])

        if split == "train":
            train_samples.append(s)
        elif split == "dev":
            dev_samples.append(s)
        elif split == "test":
            test_samples.append(s)

    print("  Before balancing:")
    print(f"    Train: {len(train_samples)}")
    print(f"    Dev: {len(dev_samples)}")
    print(f"    Test: {len(test_samples)}")

    print("\nBalancing test/dev...")

    def count_categories(samples):
        ans = [s for s in samples if s["_category"] == "answer"]
        abs_ = [s for s in samples if s["_category"] == "abstain"]
        return len(ans), len(abs_)

    dev_ans_count, dev_abs_count = count_categories(dev_samples)
    test_ans_count, test_abs_count = count_categories(test_samples)

    print(f"  Dev before: answer={dev_ans_count}, abstain={dev_abs_count}")
    print(f"  Test before: answer={test_ans_count}, abstain={test_abs_count}")

    dev_target = min(dev_ans_count, dev_abs_count)
    test_target = min(test_ans_count, test_abs_count)

    dev_answer_selected, dev_abstain_selected = balanced_sample(
        dev_samples, dev_target, dev_target, keep_all_t6=True
    )
    dev_balanced = dev_answer_selected + dev_abstain_selected
    dev_unused = [s for s in dev_samples if s not in dev_balanced]

    test_answer_selected, test_abstain_selected = balanced_sample(
        test_samples, test_target, test_target, keep_all_t6=True
    )
    test_balanced = test_answer_selected + test_abstain_selected
    test_unused = [s for s in test_samples if s not in test_balanced]

    print(
        f"  Dev after: answer={len(dev_answer_selected)}, "
        f"abstain={len(dev_abstain_selected)} (unused: {len(dev_unused)})"
    )
    print(
        f"  Test after: answer={len(test_answer_selected)}, "
        f"abstain={len(test_abstain_selected)} (unused: {len(test_unused)})"
    )
    print(f"  Train (unchanged): {len(train_samples)}")

    print("\nLeakage check...")
    dev_all = dev_balanced + dev_unused
    test_all = test_balanced + test_unused

    train_ids = set(s["source_image_id"] for s in train_samples)
    dev_ids = set(s["source_image_id"] for s in dev_all)
    test_ids = set(s["source_image_id"] for s in test_all)

    train_dev_intersection = train_ids & dev_ids
    train_test_intersection = train_ids & test_ids
    dev_test_intersection = dev_ids & test_ids

    print(f"  Train ∩ Dev: {len(train_dev_intersection)}")
    print(f"  Train ∩ Test: {len(train_test_intersection)}")
    print(f"  Dev ∩ Test: {len(dev_test_intersection)}")

    if train_dev_intersection or train_test_intersection or dev_test_intersection:
        print("\n*** LEAKAGE DETECTED! ABORTING! ***")
        raise RuntimeError("Leakage detected between splits!")

    print("\nWriting output files...")

    def write_samples(samples, filename, split_name):
        """Write the samples with an explicit `split` field, rather than inferring it from the filename.

        """
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w") as f:
            for s in samples:
                s_out = {k: v for k, v in s.items() if k != "_category"}
                s_out["split"] = split_name
                f.write(json.dumps(s_out, ensure_ascii=False) + "\n")
        return filepath

    write_samples(train_samples, "train.jsonl", "train")
    write_samples(dev_balanced, "dev.jsonl", "dev")
    write_samples(test_balanced, "test.jsonl", "test")

    print(f"  train.jsonl: {len(train_samples)}")
    print(f"  dev.jsonl: {len(dev_balanced)}")
    print(f"  test.jsonl: {len(test_balanced)}")

    print("\nGenerating split_manifest.json...")

    def compute_stats(samples, name):
        total = len(samples)
        answer_count = sum(1 for s in samples if s["_category"] == "answer")
        abstain_count = sum(1 for s in samples if s["_category"] == "abstain")

        wrong_object = sum(
            1 for s in samples if s["_category"] == "abstain" and s.get("reason") == "wrong_object"
        )
        no_target = sum(
            1 for s in samples if s["_category"] == "abstain" and s.get("reason") == "no_target"
        )

        relation_dist = defaultdict(int)
        for s in samples:
            r = s.get("metadata", {}).get("relation", "unknown")
            relation_dist[r] += 1

        crop_dist = defaultdict(int)
        for s in samples:
            crop = extract_crop(s)
            crop_dist[crop] += 1

        source_ids = set(s["source_image_id"] for s in samples)

        t6_count = sum(1 for s in samples if s.get("metadata", {}).get("relation") == "T6")

        return {
            "name": name,
            "total": total,
            "answer_count": answer_count,
            "abstain_count": abstain_count,
            "answer_ratio": round(answer_count / total, 4) if total > 0 else 0,
            "abstain_ratio": round(abstain_count / total, 4) if total > 0 else 0,
            "abstain_breakdown": {
                "wrong_object": wrong_object,
                "no_target": no_target,
            },
            "relation_distribution": dict(relation_dist),
            "crop_distribution": dict(crop_dist),
            "source_image_id_count": len(source_ids),
            "t6_count": t6_count,
        }

    manifest = {
        "train": compute_stats(train_samples, "train"),
        "dev": compute_stats(dev_balanced, "dev"),
        "dev_all_samples": compute_stats(dev_all, "dev_all (including unused)"),
        "test": compute_stats(test_balanced, "test"),
        "test_all_samples": compute_stats(test_all, "test_all (including unused)"),
        "dose_pool": {
            "name": "dose_pool",
            "total": len(fuzzy_samples),
            "note": "Fuzzy samples excluded from train/dev/test",
        },
        "leakage_check": {
            "train_dev_intersection": len(train_dev_intersection),
            "train_test_intersection": len(train_test_intersection),
            "dev_test_intersection": len(dev_test_intersection),
        },
    }

    manifest_path = os.path.join(OUTPUT_DIR, "split_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("  split_manifest.json written")

    print("\n" + "=" * 60)
    print("SPLIT REPORT")
    print("=" * 60)

    print("\n1. 四个文件的样本数:")
    print(f"   train.jsonl: {len(train_samples)}")
    print(f"   dev.jsonl: {len(dev_balanced)}")
    print(f"   test.jsonl: {len(test_balanced)}")
    print(f"   dose_pool.jsonl: {len(fuzzy_samples)}")

    print("\n2. split_manifest.json 内容:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n3. 泄漏自检结果:")
    print(f"   Train ∩ Dev: {len(train_dev_intersection)}")
    print(f"   Train ∩ Test: {len(train_test_intersection)}")
    print(f"   Dev ∩ Test: {len(dev_test_intersection)}")
    if train_dev_intersection or train_test_intersection or dev_test_intersection:
        print("   *** LEAKAGE DETECTED! ***")
    else:
        print("   All intersections are 0. PASS!")

    print("\n4. Test/Dev 均衡情况:")
    dev_ans = sum(1 for s in dev_balanced if s["_category"] == "answer")
    dev_abs = sum(1 for s in dev_balanced if s["_category"] == "abstain")
    test_ans = sum(1 for s in test_balanced if s["_category"] == "answer")
    test_abs = sum(1 for s in test_balanced if s["_category"] == "abstain")
    dev_t6 = sum(1 for s in dev_balanced if s.get("metadata", {}).get("relation") == "T6")
    test_t6 = sum(1 for s in test_balanced if s.get("metadata", {}).get("relation") == "T6")

    print(
        f"   Dev: answer={dev_ans}, abstain={dev_abs}, "
        f"ratio={dev_ans}/{dev_abs if dev_abs > 0 else 1:.4f}"
    )
    print(f"   Dev T6 count: {dev_t6}")
    print(
        f"   Test: answer={test_ans}, abstain={test_abs}, "
        f"ratio={test_ans}/{test_abs if test_abs > 0 else 1:.4f}"
    )
    print(f"   Test T6 count: {test_t6}")

    print("\nDone!")


if __name__ == "__main__":
    main()
