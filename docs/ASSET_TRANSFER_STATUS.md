# 最终资产上传核验

最终范围：2,777个文件，4,508,423,995 bytes（约4.51 GB）。权重包括论文原主方法/Independent/margin+KL选定checkpoint、三次完整重训的6个选定checkpoint，以及审稿CPU实验的最终head。清单见FINAL_ASSETS。

排除：中间epoch权重、optimizer、RNG、global_step目录、校准/hidden feature缓存、论文文件与交付快照。

不可下载数据包括：20,940个派生文件压缩为单个derived_images.tar.gz（436 MB级）、870张精确PlantDoc/PlantWild图片、项目加工标注、RefCOCOg元数据及必要原始审计结果。公开Qwen基座与COCO原图按ASSET_RECOVERY下载。

已完成的远端通道验证：

- GitHub LFS成功上传实际9 MB CPU head。
- 使用独立空LFS缓存从远端下载同head，大小8,955,227 bytes，SHA-256与原文件完全一致。
- 原论文checkpoint-444 adapter与calibrated head两个对象共约118 MB已上传成功。

整体上传状态以随后Git提交及LFS推送结果为准；通道验证不等同于整份白名单已经完整上传。
