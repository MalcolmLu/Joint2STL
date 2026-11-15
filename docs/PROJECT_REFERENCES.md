# 开源项目参考与采用

本页区分可借鉴的处理思路和已落地的功能。当前软件仍是独立的本地 Python 工作台。

| 项目 | 有关内容 | 当前采用方式 |
|---|---|---|
| [SlicerSegmentHumanBody](https://github.com/mazurowski-lab/SlicerSegmentHumanBody) | 自动生成前景；二维 / 三维连通区域标签归属；提示点与切片交互 | 增加“归属仅当前切片”选项，可在轴、冠状重排和矢状重排中局部归属，保留相邻切片；原三维归属继续可用。使用本项目自己的连通域实现。 |
| [NIAID dicom2stl](https://github.com/niaid/dicom2stl) | 物理顺序读取；表面提取、清理、平滑、减面、文件写出组成独立阶段 | 参照分阶段结构，保持已有的 DICOM 物理坐标与导出回读检查。精细预览与导出复用同一原始分辨率表面，并在掩膜变化后重新生成。 |
| [DicomToMesh](https://github.com/eidelen/DicomToMesh) | VTK 网格清理、连通小物体处理、平滑、减面与坐标处理 | 优先复用已生成的精细表面提高效率。小块阈值采用物理体积；按骨输出时保留共同的患者坐标。导出保留原始网格分辨率。 |

## 数据类型与精度边界

SlicerSegmentHumanBody 中集成的 [SegmentAnyBone](https://github.com/mazurowski-lab/SegmentAnyBone) 主模型面向 MRI。最初的 CT 版本仅参考交互和标签操作，没有把 MRI 骨权重作为 CT 骨分割的验证结果。0.4.0 新增独立 MRI 路径，采用 SegmentAnyBone 的二值骨模型、固定权重和必要推理层，并独立保留许可及修改来源；MRI 不运行 CT HU 阈值或皮质补全。

NIAID 项目的 README 描述默认缩小体数据至最大边长 256、25 次平滑与 90% 减面。这些是其流程的默认参数。本项目的精度目标使用原始体素和共同坐标，预览、导出共享精细表面；平滑参数由用户控制，导出记录相关参数。重排和插值不会增加真实影像分辨率。

DicomToMesh 支持单独居中等变换。本项目的不同骨 STL 使用同一个 DICOM LPS 空间，以保存其相对位置。

## 来源与许可记录

- `SlicerSegmentHumanBody/SegmentHumanBody/SegmentHumanBody.py`：CT 阶段查看的 blob 为 `f3c38a2698074d068e2675b534deee16a314ae0c`，当时仅采用操作思路。MRI 阶段使用其固定提交 `5f026279291d2980a01274a0d0aaa6c1a4d68960` 镜像中的官方 `bone_sam.pth`。项目及 MRI 模型声明 CC BY-NC 4.0。
- `SegmentAnyBone/README.md`：查看的 blob 为 `a5b2b1846ce5cdf8c61ac09b5d4420affa4d031e`，说明模型面向 MRI。
- `niaid/dicom2stl/dicom2stl/Dicom2STL.py`：blob `45a70e1506ddf436055620556210a203e4b1d8e2`；`utils/vtkutils.py`：blob `62927ea357b07b4893ac8a164516dd720d59f8be`。README 声明 Apache-2.0。
- `eidelen/DicomToMesh/lib/src/meshRoutines.cpp`：blob `1ada9595eef3b773651cf49be53ff346301b766a`，项目声明 MIT。

0.4.0 的 MRI 推理层来自独立的 `mazurowski-lab/SegmentAnyBone` 固定提交 `782b22140c17913317cbbbbf392deda14cfbbe4a`，必要模块位于 `vendor/segment_any_bone`，修改与来源见 `PROVENANCE.json`，权重 SHA256 见 `models/mri_model_manifest.json`。该模型只生成骨 / 背景，骨名是独立几何候选，ACL / PCL 仍使用提示辅助及人工核对。没有复制 dicom2stl 或 DicomToMesh 的实现代码。LiteMedSAM 的来源、许可与校验记录继续单独保留。
