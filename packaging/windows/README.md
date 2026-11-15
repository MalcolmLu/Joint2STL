# 构建 Windows x64 离线安装包

这个目录保留 Joint2STL 0.4.4 的 Inno Setup 配置和打包脚本。安装包使用自己的 Python 运行库，在当前用户目录安装；不会将 DICOM、CT/MRI 工程、STL 或验证资料放进安装包。

构建机需要 64 位 Python 3.12.4 或更新的 3.12 版本，以及支持大于 2 GB 安装文件的 Inno Setup。本机实测组合为 Python 3.12.14 和 Inno Setup 6.7.4。请使用独立的 `.venv`，避免把其他项目的依赖带入安装包。

1. 在源码根目录运行 `Install_NVIDIA_Windows.cmd`，安装 CUDA 版 PyTorch、CuPy 和模型依赖，并取得三份按 SHA‑256 校验的模型权重。此步骤需要联网。部署时不需要另装 Python 或完整的 CUDA Toolkit；用 CUDA 仍须有兼容的 NVIDIA 驱动。
2. 安装 [Inno Setup](https://jrsoftware.org/inno-setup/downloads)，确保编译器的 `Languages` 目录包含 [ChineseSimplified.isl](https://jrsoftware.org/inno-setup/translations)。预留足够空间生成约 8 GB 的暂存树和数 GB 的成品。
3. 在源码根目录执行以下 PowerShell 命令。把 `$iscc` 改成实际的 `ISCC.exe` 路径。

```powershell
$src = (Get-Location).Path
$python = (Resolve-Path '.venv/Scripts/python.exe').Path
$iscc = 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
$recipe = Join-Path $src 'packaging/windows'
$build = Join-Path $src ('build/windows-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path $build -ErrorAction Stop | Out-Null
Copy-Item -Path (Join-Path $recipe '*') -Destination $build
$base = & $python -I -c 'import sys; print(sys.base_prefix)'
& $python -I -B -X utf8 (Join-Path $recipe 'stage_installer.py') --build $build --source $src --base-python $base
if ($LASTEXITCODE -ne 0) { throw '构建暂存目录失败' }
& $iscc (Join-Path $build 'Joint2STL.iss')
if ($LASTEXITCODE -ne 0) { throw '安装包编译失败' }
```

检查 `$build\packages` 下的安装器，并保存 SHA‑256。验收时另选一个安装目录，检查开始菜单启动、示例工程回读、CUDA 表面滤波、CPU 以及卸载。需要验证中文路径时，可让测试进程的临时目录也包含中文。
