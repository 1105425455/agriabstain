# PlantWild OOD Benchmark 构造说明

## 口径

本 benchmark 是 PlantVillage OOD 的"单变量替换"版本：用 PlantWild 的野外真实图像替换 PlantVillage 的实验室白底图像，保持其它构造维度完全一致（13 类、per_class=12、seed=42、纯 OOD_absent+OOD_confusion、无 answerable 探针）。

### 类别映射（PlantWild → 项目统一名称）

| PlantWild 名称 | 项目统一名称 |
|---|---|
| apple scab | apple scab leaf |
| apple rust | apple rust leaf |
| corn gray leaf spot | corn gray leaf spot |
| corn rust | corn rust leaf |
| corn northern leaf blight | corn leaf blight |
| potato late blight | potato leaf late blight |
| squash powdery mildew | squash powdery mildew leaf |
| tomato bacterial leaf spot | tomato leaf bacterial spot |
| tomato early blight | tomato early blight leaf |
| tomato late blight | tomato leaf late blight |
| tomato leaf mold | tomato mold leaf |
| tomato septoria leaf spot | tomato septoria leaf spot |
| tomato yellow leaf curl virus | tomato leaf yellow virus |

### 探针构造

- **OOD_absent**（156 条）：query = 另一作物病害 → GT abstain no_target
- **OOD_confusion**（132 条）：query = 同一作物另一病害 → GT abstain wrong_object
  - apple（2 类）：24 条
  - corn（3 类）：36 条
  - tomato（6 类）：72 条
- **合计**：288 条，156 张唯一图片

## 去重策略

### A. PlantWild 内部去重
1. 文件名含 ` - Copy` 的直接排除
2. 人工质量审查排除（2 张不合格图像：论文4宫格图、带水印果实图，通过 `MANUAL_EXCLUDE` 记录）
3. MD5 精确去重（逐类内）
4. 64-bit dHash 感知去重（同类内，Hamming distance ≤ 5 视为重复，优先保留分辨率更高者）

### B. 人工质量审查排除（新增）
- Reviewer 对生成最终图片进行人工审查，发现 2 张不合规图像：
  - `corn rust / google_0181.jpg`：论文4宫格严重度分级图，带文字和标尺，非自然叶片病害图
  - `apple scab / 71.jpg`：苹果果实图，带 BIGSTOCK 水印，非叶片病害主场景
- 在构造脚本中通过 `MANUAL_EXCLUDE` 集合显式记录，在候选收集阶段排除
- 排除后从候选池顺延补齐，保持每类 12 张、总数 156 张不变

### C. 与 PlantDoc 交叉去重
- 对 PlantDoc 全部 2,567 张源图（train/val/test）预计算 dHash
- 对 PlantWild 候选（全量，先于抽样）逐张计算 dHash 并与 PlantDoc 比较
- Hamming distance ≤ 5：直接排除
- Hamming distance 6-8：记入 suspicious 审计，默认排除（保守防泄漏）
- PlantDoc 图像经过 Roboflow 导出可能已重采样/格式转换，dHash 仍可能漏检和误报

### 去重统计

| 指标 | 数值 |
|---|---|---|
| - Copy 文件排除 | 18 张 |
| 人工质量审查排除 | 2 张 |
| MD5 精确去重排除 | 1 张 |
| dHash 内部感知去重排除 | 0 张 |
| PlantDoc 交叉去重排除 (h≤5) | 27 张 |
| PlantDoc 交叉去重可疑 (h=6-8) | 0 张 |
| 内部去重后候选 | 312 张 |
| PlantDoc 去重后候选 | 285 张 |
| 最终抽样 | 156 张（13 类 × 12） |

## 残余风险

1. **PlantDoc 交叉去重局限性**：dHash 对 resize/裁剪/颜色变换不完全鲁棒，极少数 PlantDoc 图像经过大幅变换后可能未被检出。但 PlantDoc 的命名模式（Roboflow UUID）与 PlantWild（google_* / 纯数字）完全不同，且 MD5 子集抽查无命中，重叠风险为低～中。
2. **PlantWild 来源单一**：全部来自 Google Images，可能存在搜索偏好偏差。
3. **低分辨率图像**：部分图像分辨率较低（<400px），可能影响模型识别。
4. **标注质量依赖 PlantWild 论文声称的"专家精修"**，未做独立验证。

## 复现命令

```bash
cd /user/huangyongbin/newrepo/agriabstain/new_reason
python scripts/build_ood_plantwild.py
```

## 输出文件

| 文件 | 路径 |
|---|---|
| 构造脚本 | `scripts/build_ood_plantwild.py` |
| 图片 | `outputs/ood_candidates/plantwild_benchmark/images/` |
| JSONL | `outputs/eval_probes/ood_plantwild.jsonl` |
| 统计 | `outputs/ood_candidates/plantwild_benchmark/build_stats.json` |
| 去重审计 | `outputs/ood_candidates/plantwild_benchmark/dedup_audit.json` |
| 本说明 | `outputs/ood_candidates/plantwild_benchmark/README.md` |