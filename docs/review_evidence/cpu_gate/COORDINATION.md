# 运行协调

主agent已启动并完成 seed7 两组。主agent随后启动 seed21 两组，日志为 `/tmp/opencode/review-gate-grouped21.log` 与 `review-gate-independent21.log`。请先用 ps 与输出manifest确认，不重复启动同一seed。负责gate子agent接续seed42及评分即可。

更新：seed21两组已完成，主agent已启动seed42两组，日志 `/tmp/opencode/review-gate-{grouped42,independent42}.log`。不要再次启动，待完成后评分即可。Grouped models目录三个核心文件(adapter/config/calibrated head)SHA-256与原论文checkpoint444完全相同，主agent已核验。
