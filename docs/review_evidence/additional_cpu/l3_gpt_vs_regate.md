## L3 GPT predictions vs ReGate head probabilities — 相邻可见度 baseline

分组与档位完全沿用 `analyze_evidence.analyze_l3` 的 `stable_group_key` 规则（`L3-{img_prefix}__{source}`，按同图同 source_instance 聚合，作业按 `achieved_visible_ratio` 排序），不按 sample_id 简单拼接。共 183 组、567 个相邻档位对，无 group 解析失败或 duplicate-ratio 失效。

| achieved bin | GPT binary answer rate | ReGate binary argmax answer rate | ReGate mean p_answer |
|---:|---:|---:|---:|
| 0.0 | 0.3378 | 0.1757 | 0.1872 |
| 0.25 | 0.7363 | 0.5000 | 0.5044 |
| 0.5 | 0.8444 | 0.9037 | 0.8993 |
| 0.75 | 0.8824 | 0.9706 | 0.9671 |
| 1.0 | 0.9235 | 0.9891 | 0.9828 |

相邻档位违例（单调性）按 group 计算，ties 不违背：相邻较高可见度必须不低于较低可见度才视为满足。GPT 只有二元 decision，因此其“binary adjacent-decline rate”与 ReGate 的连续 `p_answer` violation rate 是不同对象，二者数值不可直接等价；为公平比较，ReGate 额外计算 binary argmax 的相邻下降比例。

| 指标（183 组 pooled） | 值 |
|---|---:|
| GPT binary adjacent-decline rate（违例对/全部相邻对） | 0.0406（23/567） |
| GPT binary adjacent-decline rate（group mean） | 0.0419 |
| ReGate p_answer adjacent violation rate（同原分析） | 0.2152（122/567） |
| ReGate p_answer group mean violation（recompute） | 0.2145（现分析 0.2145） |
| ReGate binary argmax adjacent-decline rate | 0.0088（5/567） |

Endpoint（组内存在精确 1.0 与 0.0 档，148 组）：GPT binary endpoint decline=0.0338，ReGate binary endpoint decline=0.0000，ReGate p_answer endpoint delta mean=0.8044（现分析 0.8044）。

> 注意：二元决策单调性与连续 `p_answer` 单调性衡量不同的性质；同一组数据上二者数值不同属预期，不能用 GPT 的二元下降比例直接对标 ReGate 的连续 violation rate。相关 JSON：`l3_gpt_vs_regate.json`。
