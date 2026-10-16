# CPU 补充实验协议

本轮按用户要求完成无需 GPU 的分析和小型 head 拟合。原始数据、模型、论文与正式结果保持原样；新增结果分别保存，不覆盖论文基线。

## Gate 稳定性

- 原论文 Grouped checkpoint-444 与 Independent checkpoint-222 分别固定，不重新训练 backbone。
- 训练输入使用 `papermodel_lineargate/runs/{grouped,independent}/calibration` 的缓存表征；读取 manifest、shard 数量、标签分布及来源，禁止将同名的其他重训 checkpoint-444 混入。
- 复制原有 `papermodel_lineargate/scripts/fit_gates.py`：MLP 4096→512→256→3，GELU、dropout 0.1，train-feature mean/std，full-batch Adam lr=1e-3、weight_decay=1e-3、无类别权重，最多1,000 epoch，每50 epoch评估，patience=5，以最低dev loss选模型。
- 两组各拟合 initialization seed 7、21、42。全部运行均纳入汇总；不依据test表现挑选seed或调参。这是同一backbone下的head-fit稳定性实验，不增加完整模型训练seed数。
- 所有 tensor 保持CPU，`CUDA_VISIBLE_DEVICES=''`，每进程最多4个计算线程、最多两个同类拟合并行。
- 测试使用已保存的 `head_hidden` 与 `answer_candidate`。新模型的三类概率全部注入打分记录后计算answer与reason，避免新answer mask搭配旧reason概率。
- 发布头的mu/sd与缓存重算mu/sd单独比较；差异只能证明不匹配，不能直接判定它来自某个其他checkpoint。

## L2

- 从原图尺寸、源标注bbox和实际crop目标恢复零档裁剪区域；对query类别的全部已标注实例计算与裁图的交集。
- 区分完全保留、部分残留、无已标注实例残留及缺信息。无已标注残留仍依赖原标注完整性。
- 诊断筛选只依赖标注与几何，不用预测决定保留；分别报告原395条结果和诊断子集结果。原标签不覆盖。
- 原样本顺序与ID均核对，避免重复sample ID被dict覆盖。与L3进行规范化image/query/source对应检查。

## 其他补充

- Frozen-base缓存的CPU head实验单独记录train/dev/test与候选生成来源；使用Base native候选，禁止使用adapted候选冒充frozen系统。
- 已有完整多seed训练结果只作历史实验复核，不称为本轮新训练；逐seed检查原始metrics及所选checkpoint。
- 标注审计仅生成可交给两名标注者的分层样本和空白记录；没有人工标签时不产生人审正确率。
- GPT/SFT confidence、latency、最近方法和全24类结果仅在已有记录足以支持时统计；不以其他模型概率或不匹配运行补值。
