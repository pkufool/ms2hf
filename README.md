# ms2hf

`ms2hf` 是一个用于在 [ModelScope 魔搭](https://modelscope.cn/) 和 [Hugging Face](https://huggingface.co/) 之间同步模型 / 数据集仓库的命令行工具。

它主要面向中文用户，尤其适合在 Google Colab、Gitpod、云服务器临时环境等「能访问 Hugging Face 和 ModelScope，但本地磁盘空间有限」的场景中使用。

和一次性 clone / 下载整个仓库不同，`ms2hf` 会按文件逐个执行：

1. 从源平台下载一个文件；
2. 上传到目标平台；
3. 上传成功后删除本地临时文件；
4. 继续处理下一个文件。

这样可以显著降低同步大模型或大数据集时对本地存储空间的要求。

## 功能特性

- 提供两个命令：
  - `ms2hf`：从 ModelScope 同步到 Hugging Face；
  - `hf2ms`：从 Hugging Face 同步到 ModelScope。
- 同时支持模型仓库和数据集仓库。
- 支持从环境变量读取 token：
  - `HF_TOKEN`
  - `MS_TOKEN`
- 也支持通过命令行参数传入 token。
- 默认逐个文件同步，适合低磁盘空间环境。
- 默认会检查目标仓库是否已有同路径文件，已存在则跳过。
- 支持 `--force` 强制重新上传已有文件。
- 支持 `--include-regex` 只同步匹配的文件。
- 支持 `--exclude-regex` 排除指定文件或文件夹。
- 支持 `--workers` 并发同步，提高传输效率。
- 支持 `--dry-run` 预览将要同步的文件，不实际下载 / 上传。
- 支持 `--create-target` 自动创建目标仓库。
- 同步过程中显示进度条。
- 同步完成后输出成功、跳过、失败数量。

## 安装

### 从 PyPI 安装

项目发布到 PyPI 后，可以直接安装：

```bash
pip install ms2hf
```

安装完成后会得到两个命令：

```bash
ms2hf --help
hf2ms --help
```

### 从本地源码安装

在项目根目录执行：

```bash
pip install .
```

## Token 配置

### 方式一：通过环境变量配置，推荐

```bash
export HF_TOKEN="hf_xxx"
export MS_TOKEN="xxx"
```

然后直接运行同步命令：

```bash
ms2hf k2-fsa/OpenDialog your-hf-name/OpenDialog --repo-type dataset
```

### 方式二：通过命令行参数传入

```bash
ms2hf k2-fsa/OpenDialog your-hf-name/OpenDialog \
  --repo-type dataset \
  --hf-token "hf_xxx" \
  --ms-token "xxx"
```

> 注意：命令行参数可能会出现在 shell history 或进程列表中。如果在共享环境中使用，建议优先使用环境变量。

## 基本用法

命令格式：

```bash
ms2hf SOURCE_REPO TARGET_REPO [OPTIONS]
hf2ms SOURCE_REPO TARGET_REPO [OPTIONS]
```

其中：

- `SOURCE_REPO` 是源仓库 ID，例如 `k2-fsa/OpenDialog`；
- `TARGET_REPO` 是目标仓库 ID，例如 `your-name/OpenDialog`；
- 仓库类型通过 `--repo-type` 指定，可选值为：
  - `model`
  - `dataset`

## 使用示例

### 1. 从 ModelScope 同步数据集到 Hugging Face

```bash
ms2hf k2-fsa/OpenDialog your-hf-name/OpenDialog \
  --repo-type dataset \
  --source-revision master \
  --create-target \
  --work-dir /content/dataset_sync
```

说明：

- 从 ModelScope 的 `k2-fsa/OpenDialog` 数据集读取文件；
- 同步到 Hugging Face 的 `your-hf-name/OpenDialog` 数据集；
- 如果目标仓库不存在，使用 `--create-target` 创建；
- 临时文件存放在 `/content/dataset_sync`；
- 上传成功后默认删除本地临时文件。

### 2. 从 Hugging Face 同步模型到 ModelScope

```bash
hf2ms bert-base-uncased your-ms-name/bert-base-uncased \
  --repo-type model \
  --create-target
```

### 3. 只同步指定类型的文件

例如只同步 `data/` 目录下的 parquet 文件：

```bash
ms2hf org/big-dataset your-name/big-dataset \
  --repo-type dataset \
  --include-regex '^data/.*\.parquet$'
```

### 4. 排除指定目录

例如排除所有 `tmp` 目录：

```bash
ms2hf org/big-dataset your-name/big-dataset \
  --repo-type dataset \
  --exclude-regex '(^|/)tmp/'
```

### 5. 同时使用 include 和 exclude

```bash
ms2hf org/big-dataset your-name/big-dataset \
  --repo-type dataset \
  --include-regex '^data/' \
  --exclude-regex '(^|/)cache/' \
  --exclude-regex '\.tmp$'
```

规则：

- 如果指定了 `--include-regex`，只有匹配 include 的文件会被考虑；
- 如果文件同时匹配 `--exclude-regex`，会被排除；
- 两个参数都可以重复使用。

### 6. 强制覆盖 / 重新上传

默认情况下，如果目标仓库已经存在同路径文件，工具会跳过该文件。

如果你希望重新上传，可以使用：

```bash
ms2hf org/model your-name/model \
  --repo-type model \
  --force
```

### 7. 预览同步计划，不实际执行

```bash
ms2hf org/dataset your-name/dataset \
  --repo-type dataset \
  --dry-run
```

这会列出将要同步、跳过的文件统计，但不会下载或上传任何文件。

### 8. 提高并发数

默认 `--workers 1`，也就是一次只处理一个文件，这样最省磁盘空间。

如果环境磁盘空间足够，可以提高并发数：

```bash
ms2hf org/dataset your-name/dataset \
  --repo-type dataset \
  --workers 4
```

> 注意：`--workers` 越大，同时保存在本地的临时文件越多，磁盘占用也可能越高。

### 9. 保留本地下载文件

默认上传成功后会删除本地临时文件。如果你希望保留下载的文件用于调试，可以使用：

```bash
ms2hf org/dataset your-name/dataset \
  --repo-type dataset \
  --keep-local
```

## 常用参数说明

| 参数 | 说明 |
| --- | --- |
| `source_repo` | 源仓库 ID，例如 `k2-fsa/OpenDialog` |
| `target_repo` | 目标仓库 ID，例如 `your-name/OpenDialog` |
| `--repo-type` / `--type` | 仓库类型，`model` 或 `dataset`，默认 `model` |
| `--source-revision` | 源仓库 revision，可以是分支、tag 或 commit |
| `--target-revision` | 目标仓库 revision；是否生效取决于目标平台 SDK 支持情况 |
| `--work-dir` | 本地工作目录，默认当前目录 |
| `--include-regex` | 只同步匹配该正则的文件，可重复指定 |
| `--exclude-regex` | 排除匹配该正则的文件，可重复指定 |
| `--force` | 即使目标已有同路径文件，也重新上传 |
| `--workers` | 并发同步文件数，默认 `1` |
| `--max-retries` | 每个下载 / 上传步骤的最大重试次数，默认 `3` |
| `--retry-sleep` | 重试间隔秒数，默认 `10` |
| `--hf-token` | Hugging Face token；默认读取 `HF_TOKEN` 环境变量 |
| `--ms-token` | ModelScope token；默认读取 `MS_TOKEN` 环境变量 |
| `--create-target` | 如果目标仓库不存在，尝试自动创建 |
| `--private` | 和 `--create-target` 一起使用，创建私有仓库 |
| `--dry-run` | 只预览，不实际同步 |
| `--keep-local` | 上传后保留本地临时文件 |
| `--commit-message-template` | 自定义每个文件上传时的 commit message 模板 |

## 在 Google Colab 中使用

示例：

```python
!pip install ms2hf
```

```python
import os
os.environ["HF_TOKEN"] = "hf_xxx"
os.environ["MS_TOKEN"] = "xxx"
```

```python
!ms2hf k2-fsa/OpenDialog your-hf-name/OpenDialog \
  --repo-type dataset \
  --create-target \
  --work-dir /content/dataset_sync
```

## 在 Gitpod / 云服务器中使用

```bash
pip install ms2hf

export HF_TOKEN="hf_xxx"
export MS_TOKEN="xxx"

ms2hf k2-fsa/OpenDialog your-hf-name/OpenDialog \
  --repo-type dataset \
  --create-target \
  --work-dir ./dataset_sync
```

## 同步策略说明

### 为什么逐个文件同步？

很多模型和数据集非常大，如果一次性下载整个仓库，很容易耗尽 Colab、Gitpod 或临时云服务器的磁盘空间。

`ms2hf` 的策略是：

```text
下载单个文件 -> 上传单个文件 -> 删除本地文件 -> 处理下一个文件
```

这样本地磁盘通常只需要容纳一个文件，或在开启多线程时容纳少量文件。

### 如何判断文件是否已经同步？

目前工具默认按“目标仓库是否存在同路径文件”判断：

- 如果目标仓库已有同路径文件：跳过；
- 如果目标仓库没有该路径：下载并上传；
- 如果指定 `--force`：不检查跳过，直接重新上传。

这个策略比较保守，能避免重复传输，但不会跨平台比较 hash。如果你怀疑目标文件内容不是最新的，请使用 `--force`。

## 当前限制

- 目前“是否已存在”主要按文件路径判断，不做跨平台 hash 对比。
- 每个文件会单独上传并产生一次提交，适合低存储环境，但同步大量小文件时可能较慢。
- ModelScope 当前 SDK 的 `upload_file` 通常提交到默认分支，因此 `--target-revision` 在目标为 ModelScope 时可能不会生效。
- 不同平台对模型 / 数据集仓库的可见性、权限、分支、LFS 策略支持不完全一致，具体行为以平台 API 为准。

## 开发者说明

本项目的入口定义在 `pyproject.toml`：

```toml
[project.scripts]
ms2hf = "ms2hf:main_ms2hf"
hf2ms = "ms2hf:main_hf2ms"
```

本地检查：

```bash
python -m py_compile ms2hf.py
python ms2hf.py --help
```

本地安装测试：

```bash
pip install -e .
ms2hf --help
hf2ms --help
```

## 许可证

Apache-2.0

## GitHub Actions

本项目内置了两个 GitHub Actions workflow，分别用于发布 PyPI 和在线触发同步任务。

### 1. 发布到 PyPI

Workflow 文件：

```text
.github/workflows/publish-pypi.yml
```

触发方式：

1. 推送版本 tag 到 GitHub，例如：

```bash
git tag v0.1.0
git push origin v0.1.0
```

2. 或者在 GitHub 网页端手动运行：

```text
Actions -> Publish to PyPI -> Run workflow
```

该 workflow 会自动：

- 检出代码；
- 安装构建工具；
- 执行 `python -m build`；
- 执行 `twine check`；
- 使用 `twine upload` 发布到 PyPI。

发布使用 PyPI API Token，不使用 Trusted Publisher。请先在 GitHub 仓库 Secrets 中配置：

```text
Settings -> Secrets and variables -> Actions -> New repository secret
```

Secret 名称必须是：

```text
PYPI_TOKEN
```

Secret 值填写 PyPI 上创建的 API Token，通常以 `pypi-` 开头。

### 2. 在 GitHub 网页端手动同步模型 / 数据集

Workflow 文件：

```text
.github/workflows/sync.yml
```

使用方式：

```text
GitHub 仓库页面 -> Actions -> 同步 ModelScope / Hugging Face -> Run workflow
```

网页端会要求输入：

- 同步方向：`ms2hf` 或 `hf2ms`；
- 仓库类型：`model` 或 `dataset`；
- 源仓库 ID；
- 目标仓库 ID；
- source revision；
- target revision；
- include 正则；
- exclude 正则；
- workers；
- 是否 force；
- 是否 create target；
- 是否 dry run；
- Hugging Face token；
- ModelScope token。

推荐先在仓库 Secrets 中配置 token：

```text
Settings -> Secrets and variables -> Actions -> New repository secret
```

建议添加：

```text
HF_TOKEN
MS_TOKEN
```

配置后，每次在网页端运行同步任务时就不需要手动输入 token。

如果没有配置 Secrets，也可以在 Run workflow 页面临时输入 `hf_token` 和 `ms_token`。workflow 会对 token 做日志 mask，但 GitHub Actions 输入本身不等同于 Secret；在多人协作仓库中，仍然更推荐使用 Secrets。

#### 网页端同步示例

从 ModelScope 同步数据集到 Hugging Face：

```text
direction: ms2hf
repo_type: dataset
source_repo: k2-fsa/OpenDialog
target_repo: your-hf-name/OpenDialog
source_revision: master
workers: 1
create_target: true
force: false
dry_run: false
```

从 Hugging Face 同步模型到 ModelScope：

```text
direction: hf2ms
repo_type: model
source_repo: bert-base-uncased
target_repo: your-ms-name/bert-base-uncased
workers: 1
create_target: true
```

#### 建议先 dry run

第一次同步前，建议先勾选：

```text
dry_run: true
```

确认源文件数量、过滤规则和跳过逻辑符合预期后，再正式执行同步。
