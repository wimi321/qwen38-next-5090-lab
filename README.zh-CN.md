# Qwen3.8 Next 5090 Lab

[English](README.md) · [完整复现记录](docs/qwen4-exp.md) · [模型状态](docs/models.md) · [安全说明](SECURITY.md)

> [!IMPORTANT]
> 本项目是 [FreeToken](https://github.com/FlashML-org/FreeToken) 的**非官方、
> 实验性下游项目**，与 Qwen、RadixArk、NVIDIA、FlashML 及其贡献者均无隶属
> 或背书关系。

本项目让完整 135 GB
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
checkpoint 的**文本塔**在一张 32 GB RTX 5090 上运行。它在 FreeToken 的
OpenAI 兼容 API 之上组合了稀疏 PLE 行流式读取、QSA Triton kernel、四路
gated residual 和异构 MoE offload。

这是一个仅发布源码的开发者预览版。仓库不再分发模型权重，不占用
`freetoken` 官方包的发布名，也不声称“首个”“唯一”或“最快”。

![Qwen3.8 Next 5090 Lab 架构](docs/assets/q38lab-architecture.svg)

## 已验证的 alpha 边界

| 项目 | `v0.1.0-alpha.1` 承诺范围 |
|---|---|
| 输入 / 输出 | 仅文本输入和文本输出 |
| 硬件 | 单张 RTX 5090（32 GB）、TP=1、WSL2/Ubuntu 24.04 |
| 调度 | 同时运行一个请求，总 token 上限 8,192 |
| Cache / graph | naive cache；关闭 radix prefix reuse 和 CUDA graph |
| 专家计算 | 保留 NVFP4 packed routed weights，BF16 activation（**W4A16 compatibility**） |
| Checkpoint | 固定 revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594` 的完整 135,253,622,894 字节目录 |
| 不在范围内 | 图片、视频、音频、MTP、多并发、32K/262K、原生 W4A4 parity |

checkpoint 声明 routed expert 使用 W4A4 方案。本 alpha 保留 packed NVFP4
权重，但没有消费 checkpoint 的 activation-scale 契约，所以当前结果只能称为
W4A16 compatibility，不能继承 W4A4 的质量结论，也不能声称与其数值等价。

## 为什么能放进这台机器

- **PLE 只读取命中的行。** 约 51 GB FP8 PLE 表以只读 safetensors mmap
  形式存在。CPU 只复制命中行，以 FP32 解码并应用 scale，然后转换为 pinned
  BF16 staging，最后传到 GPU。项目没有把整个 PLE 表搬进 GPU。
- **QSA 使用独立缓存。** 12 个稀疏注意力层在 4-token block 上选择，保存主
  K/V 和 index-key 状态；其余 36 层运行 gated delta recurrence。
- **专家权重跨越多级内存。** 每层 512 个 routed experts 采用 top-10，结合
  pageable CPU layers、pinned host banks、PCIe 传输和 GPU expert cache；
  shared expert 仍在每层参与计算。
- **性能数字有原始证据。** environment、最终解析配置、请求时间、RSS、VRAM、
  page fault、PCIe 采样、测试结果和 checksum 一起保存；README 的数字由
  `summary.json` 生成。

## 从源码安装

已验证环境是 WSL2 内的 Linux x86-64。源码和模型必须放在 WSL 发行版自己的
ext4 文件系统中，不要放到 `/mnt/c` 或 `/mnt/d`。WSL 内安装 CUDA 13 toolkit，
不要安装 Linux 显卡驱动；GPU 由 Windows NVIDIA 驱动提供给 WSL。

```bash
git clone https://github.com/wimi321/qwen38-next-5090-lab.git
cd qwen38-next-5090-lab

uv sync --locked --extra accel
source .venv/bin/activate

q38lab doctor
```

为兼容现有 FreeToken 代码，源码包仍保留 `import freetoken` 和 `ft` 命令。
请使用独立虚拟环境；本包不能和上游 `freetoken` distribution 共装。

## 下载固定版本的 checkpoint

下载前请阅读[模型许可证说明](MODEL_LICENSES.md)。命令要求用户明确确认 Qwen
许可条款，固定指定 revision；如果该 revision 消失，命令会停止，不会静默换模型。

```bash
q38lab download --accept-qwen-license

# 可选：重新读取并 hash 整个本地 checkpoint。
q38lab download --accept-qwen-license --full-verify
```

已验证副本包含 419 个非缓存文件、206 个 safetensors 文件，共
135,253,622,894 字节。模型文件不会进入 Git 仓库或 GitHub Release。

固定版本下载器会在 Hugging Face 解析并下载准确 commit 后，在 checkpoint
目录外写入验证凭据。`serve` 每次都会重新核对全部控制文件 hash 和完整文件
stat fingerprint。可选的 `--full-verify` 还会重新读取全部 135 GB，发布证据
必须使用它；只有文件数量/大小正确而没有凭据的目录不会被 `serve` 接受。

## 启动服务

```bash
q38lab serve --profile rtx5090-wsl2
```

profile 固定在已审计的启动边界：`memory-ratio=0.89`，`num-tokens`、
`max-seq` 和 `max-prefill` 均为 8192，`max-running-requests=1`，naive cache，
`qsa_triton`，graph 关闭，MoE offload 和 expert cache 自动定容。配置优先级为：
CLI 参数 > `Q38LAB_*` 环境变量 > profile 默认值。

无鉴权服务默认只监听 `127.0.0.1`。如果没有显式危险确认参数，`q38lab` 会
拒绝把无鉴权服务绑定到非 loopback 地址。`--unsafe-allow-non-loopback` 只表示
风险确认，不会增加鉴权或 TLS；不要把此开发服务直接暴露到网络。

运行完整 API 冒烟测试：

```bash
q38lab smoke
```

也可以用 OpenAI Python 客户端调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1919/v1", api_key="local-only")
response = client.chat.completions.create(
    model="qwen3.8-flash-next-nvfp4",
    messages=[{"role": "user", "content": "只回复：q38lab-ok"}],
    temperature=0,
    max_tokens=32,
)
print(response.choices[0].message.content)
```

流式调用只需加入 `stream=True` 并迭代 chunks。`q38lab smoke` 还覆盖两种
thinking 模式和一次带 schema 的工具调用。模型输出不是安全边界；应用仍必须
校验工具名和参数。

## 复现实测证据

```bash
q38lab bench --out results/rtx5090-YYYY-MM-DD
```

这是唯一的正式 release harness，不是快速 microbenchmark。它会重新 hash 完整
checkpoint，运行非 slow 测试，验证 prompt/API 门槛与 100 个顺序请求，然后另行
运行至少 30 分钟的周期生成 soak，并连续采样资源。它读取 `q38lab serve` 写入的
启动证明；源码树不干净、运行 commit 不匹配或服务进程不一致都会直接失败。

不能脱离相邻的 environment 和 resolved-config 记录单独比较数字。TTFT 从客户端
观测；“effective prefill”包含固定请求开销与 CPU staging，不是纯 kernel benchmark。

首个硬件记录包括：非 slow 测试 1,454 passed / 9 skipped / 11 deselected；
8,176-token prompt 峰值显存 31,542 MiB；WSL 峰值 RSS 67.895 GiB；100/100
顺序请求成功，延迟 p50/p95 为 2.345/2.363 秒。8,176-token 请求的客户端 TTFT
p50 为 7.225 秒，effective prefill p50 为 1,130.83 tok/s。权威表格由英文
[README](README.md) 的 evidence 生成器维护。

验证时 WSL 的 `swap=0`。Windows 宿主已有 128 GiB pagefile，且系统级汇总用量
非零，因此本项目**不声称宿主没有发生分页**。早期 7-token completion 也不会被
当作稳态 decode benchmark；release gate 会另测 256–512 token decode。

引用数字前请阅读[完整步骤与限制](docs/qwen4-exp.md)。

## 项目状态

alpha 有意保持小范围，后续里程碑依次为：

1. 完整 reference parity 与原生 SM120 W4A4 activation path。
2. PLE telemetry / hot-row cache，以及经过实机验证的 CUDA graph capture/replay。
3. 图片处理器、媒体传输、vision prefill、mRoPE 和安全 cache 语义。

目前媒体输入会被拒绝。checkpoint 本身是多模态，并不意味着在线服务已经支持
多模态；processor、vision tower、placeholder expansion 和 media-aware cache
必须全部实现并验证后才能这样宣称。

## 来源、贡献与引用

本项目保留 FreeToken 的完整历史，审计基线为上游 commit
`9ef3651309fe4058672f2cc92069238dea06be1b`。下游变更见
[MODIFICATIONS.md](MODIFICATIONS.md)，保留的第三方归属见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

每项贡献都必须由人类维护者理解并实测。项目不接受模型权重、隐私日志、编造的
benchmark 或无人复核的纯 agent 提交。提交 issue 或 PR 前请阅读
[CONTRIBUTING.md](CONTRIBUTING.md)。学术使用请同时引用本项目
[CITATION.cff](CITATION.cff) 和提供底层 serving engine 的 FreeToken 工作。

## 许可证

- 仓库代码和文档：[Apache License 2.0](LICENSE)，并保留 FreeToken 和第三方归属。
- Qwen 模型文件：独立的
  [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)。
- RadixArk checkpoint：模型卡要求参照源模型条款，并称其为 candidate release。

仓库的 Apache-2.0 不授予模型权利或商标权。本说明不是法律意见；使用者必须自行
确认其使用和部署所适用的全部条款。
