# 最终资产上传核验

最终范围：2,777个文件，4,508,423,995 bytes（约4.51 GB）。权重包括论文原主方法/Independent/margin+KL选定checkpoint、三次完整重训的6个选定checkpoint，以及审稿CPU实验的最终head。清单见FINAL_ASSETS。

排除：中间epoch权重、optimizer、RNG、global_step目录、校准/hidden feature缓存、论文文件与交付快照。

不可下载数据包括：20,940个派生文件压缩为单个derived_images.tar.gz（436 MB级）、870张精确PlantDoc/PlantWild图片、项目加工标注、RefCOCOg元数据及必要原始审计结果。公开Qwen基座与COCO原图按ASSET_RECOVERY下载。

已完成的远端通道验证：

- GitHub LFS成功上传实际9 MB CPU head。
- 使用独立空LFS缓存从远端下载同head，大小8,955,227 bytes，SHA-256与原文件完全一致。
- 原论文checkpoint-444 adapter与calibrated head两个对象共约118 MB已上传成功。

最终上传已完成：Git LFS返回 `1532/1532 objects, 4.5 GB, done`，资产提交为`981882a`，已推送到GitHub main。最终清单2,777项全部纳入Git（小型标注/配置为普通Git，大文件为LFS）；本地验证无缺失、无仅pointer文件、无论文跟踪路径。远端取回已对实际head进行空缓存SHA-256验证；未对整4.5GB重新完整下载。

后续接手审查：当前`git lfs ls-files`识别1,537个LFS路径；上传日志记录1,532个对象，两者统计单位不同。更新后的验证器检查清单实体、LFS大小、pointer、缺失清单项、清单外资产和论文排除，全部无错误。恢复工具通过中断后partial目录、重复执行、缺源、LFS pointer及恶意归档路径测试。README/AGENTS/HANDOFF已统一为最终资产策略。
