# JetPack 6 上 Qwen3-ASR native vLLM 可行性

## 结论

当前 `.venv-gpu` **不能直接构建或运行 Qwen3-ASR 官方 vLLM 路线**。Qwen3-ASR
官方包固定 `vllm==0.14.0`；该 vLLM 版本固定 `torch==2.9.1`，而项目稳定环境是
Jetson-native PyTorch `2.5.0a0+872d972e41.nv24.08`。强行用当前 PyTorch 编译，
即使绕过 Python 依赖检查，也会面对 C++/CUDA ABI 与算子接口断层，不能视为可靠方案。

值得继续投入的路线是 **Jetson Containers 中的全隔离容器构建**，而不是在宿主机
`.venv-gpu` 里打补丁。它需要独立的 Jetson PyTorch 2.9.1、Triton、FlashInfer、
FlashAttention 和 vLLM 0.14.0 源码构建；成本较高，但不会替换现有机器人运行环境。

## 已核验的版本事实

| 组件 | 官方/当前要求 | 本机稳定环境 | 判断 |
|---|---|---|---|
| Qwen3-ASR Python 包 | `vllm==0.14.0` | 未安装 vLLM | 必须使用 0.14.0 API 路线 |
| vLLM 0.14.0 | `torch==2.9.1` | PyTorch 2.5 Jetson build | 不兼容，必须新建 PyTorch 栈 |
| vLLM CUDA 架构 | CMake 明确包含 8.7 | Orin SM 8.7 | 主 vLLM CUDA 扩展架构可行 |
| CUDA | 源码构建可指定现有 toolkit | CUDA 12.6 | 可行，必须 `CUDA_HOME=/usr/local/cuda` |
| 宿主工具链 | CMake ≥3.26.1；完整构建依赖 Triton 等 | CMake 3.22.1、无可用 Triton | 宿主 venv 不可直接构建 |
| Jetson Containers | 当前 recipe 支持动态 SM87 patch 与 Jetson PyTorch | Docker daemon 需特权访问 | 推荐，但本轮未启动容器构建 |

版本核验使用以下官方源码快照：

- Qwen3-ASR commit `7c6daf77a2421100f5fb066495372c00129d39ff`：
  `pyproject.toml` 的 vLLM extra 固定 0.14.0，并通过 `Qwen3ASRModel.LLM(...)`
  暴露官方接口。
- vLLM tag `v0.14.0`, commit `b17039bccc17b94a2ea107272ad7bc93508708df`：
  `pyproject.toml`、`requirements/build.txt` 和 `requirements/cuda.txt` 固定
  PyTorch 2.9.1；`CMakeLists.txt` 的 CUDA 支持架构包含 8.7。
- jetson-containers commit `70c149aea6126594153ef7690ec4890f8f396518`：
  vLLM build recipe 使用 `CUDA_HOME=/usr/local/cuda`、aarch64 低并发编译，
  并动态生成目标 SM patch；PyTorch recipe 包含 2.9.1 的 JetPack 6 变体。

对应上游资料：[Qwen3-ASR 官方仓库](https://github.com/QwenLM/Qwen3-ASR)、
[vLLM v0.14.0](https://github.com/vllm-project/vllm/tree/v0.14.0)、
[Jetson Containers](https://github.com/dusty-nv/jetson-containers) 和
[JetPack 6.2 组件说明](https://developer.nvidia.com/embedded/jetpack-sdk-62)。

## 隔离检查实际结果

本轮创建了 `/home/nvidia/Developer/vllm-orin-jp6`，并分别浅克隆三份官方源码。
新建的 `.venv-preflight` 不继承系统 site-packages，检查结果为：

```text
python 3.10.12
machine aarch64
torch_visible False
vllm_visible False
```

没有向该环境安装 generic aarch64/SBSA PyTorch wheel，也没有修改 `.venv-gpu`。
没有执行正式编译，原因不是隐藏失败，而是 preflight 已发现以下硬阻断：

1. vLLM 0.14.0 与当前 PyTorch 2.5 存在明确版本断层。
2. 本机 `torch.compile` 实验已经证明当前环境没有可工作的 Triton。
3. 宿主 CMake 3.22.1 低于 vLLM 0.14.0 的 3.26.1 构建要求。
4. 当前用户不能直接访问 Docker daemon；按“不做非必要 sudo 系统操作”的约束，
   本轮没有启动需要特权、下载大体积基础镜像并重建整套 AI 栈的容器任务。

## 推荐构建路线

建议先在 Jetson Containers 的隔离容器中补一个 `vllm:0.14.0` package variant，
让依赖解析选择 Jetson PyTorch 2.9.1，而不是复用宿主 PyTorch 2.5。关键参数应为：

```bash
LSB_RELEASE=24.04
CUDA_VERSION=12.6
PYTORCH_VERSION=2.9.1
TORCH_CUDA_ARCH_LIST=8.7
CUDA_HOME=/usr/local/cuda
MAX_JOBS=2
NVCC_THREADS=1
```

构建完成后先做三层验收，再接入本项目：

1. `torch.cuda.get_device_capability() == (8, 7)`，且 vLLM 扩展只包含/可运行 SM87。
2. `Qwen3ASRModel.LLM(model=<local-0.6B>, dtype=torch.bfloat16)` 能完成固定 WAV。
3. 用与 Transformers 完全相同的 warm-up、文本一致性和 wall-clock 口径复测，
   同时确认 CUDA Graph 实际启用；否则不能宣称 vLLM 带来收益。

## 投入判断

如果目标是把单请求 1.2 秒降到 SenseVoice 的几十毫秒，**不建议**在 JP6 上继续投入
大量时间做 native vLLM：模型架构和 batch-1 内存/调度限制仍然存在。若目标是流式输出、
多请求吞吐或验证 CUDA Graph 对 Orin 的实际收益，则隔离容器路线值得做一个时间盒验证；
成功后也应作为新的可选 runtime，而不是替换稳定 Transformers/SenseVoice 环境。
