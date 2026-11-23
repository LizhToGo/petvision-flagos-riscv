# PetVision × FlagOS（SpacemiT K1 真机）迁移验证

把结课项目 **PetVision** 的最终分类器（ResNet-50，37 类，256×256，raw FP32）
迁移到 RISC-V 真机 **SpacemiT K1 / X60**，并在同一台机器上验证 FlagGems
`spacemit` 后端的真实 JIT 执行、数值正确性与耗时。

> 原课设只读仓库：不复制代码、不写入、不改 Git 状态；模型权重只读打开，仓库内没有任何 `.pt/.pth` 或原图。

## 1. 结论（先看这里）

| 项目 | 结果 | 证据 |
| --- | --- | --- |
| 原生 PyTorch 真机前向 | ✅ 2115.76 ms/前向；53 次 `aten::conv2d` | `results/k1/native_seed0.json` |
| 与 x86 参考 logits（seed 0） | ✅ max_abs_err `1.19e-06`，Top-1 都是 `leonberger` | 同上（`checks` 全过） |
| 真图 `Abyssinian_201.jpg` | ✅ `Abyssinian`，p=`0.997643857`，max_abs_err `1.05e-05` | `results/k1/native_image.json` |
| `aten._softmax` 厂商 kernel | ✅ 数值 `5.96e-08`，真实 kernel launch 1 次 | `results/k1/ops_softmax.json` |
| 真图 → 37 类概率（仅厂商 softmax） | ✅ 1 次 `softmax_kernel_spacemit`，概率差 `1.42e-14`，Top-1 一致 | `results/k1/flag_gems_softmax_image.json` |
| 厂商 `mean.dim`（ResNet 全局池化） | K1 真机修复通过，两项回归测试通过 | [实测数据](results/k1/mean_dim_pr_9773.json) · [FlagGems PR #9773](https://github.com/flagos-ai/FlagGems/pull/9773) |
| 厂商 `conv2d` kernel | ✅ K1 真机修复；53/53 次卷积走两阶段厂商 kernel | [实测数据](results/k1/conv2d_pr_9823.json) · [FlagGems PR #9823](https://github.com/flagos-ai/FlagGems/pull/9823) |
| 通用 FlagGems `conv2d` | ✅ 数值 `9.54e-07`（旧基线，非最终路径） | `results/k1/conv2d_vendor_failure.json` |

旧版固定环境的真图结果启用厂商 `aten._softmax` 后处理；`mean.dim` 修复见 PR #9773。
Conv2d 修复版使用同一权重、同一真图在 K1 上完成 53/53 次厂商卷积验证；性能结果不声称加速。

## 2. 环境（K1 真机实测）

| 项目 | 值 |
| --- | --- |
| 硬件 | SpacemiT K1 / X60，rv64imafdcv…zvfh…**ime**，8 核，约 15 GiB 可用 |
| 系统 | Bianbu 2.3，Linux 6.6.63-cloud，Python 3.12.3 |
| venv | `$HOME/petvision-k1/venv`（`--without-pip` + 官方 `get-pip.py`） |
| PyTorch | `2.8.0+spacemit.0`（cp312/linux_riscv64） |
| torchvision | `0.23.0`（cp312/linux_riscv64） |
| Triton | `3.6.0+spacemit.a5`；`triton.__version__` 报 `3.6.0` |
| Triton 驱动 | `CPUDriver`，target `AICPUTarget(arch='spacemit-a60', core=4, ai_core=2, num_threads=2)` |
| FlagGems | 基线 commit `43c9cab402f7e286de9cbfc6238ea3e7bfe354cb`；Conv2d 修复 commit `5481b6a3e84d1a5a5bb4087cf9a9f28ae88d1335`（均为源码安装） |
| vendor | `GEMS_VENDOR=spacemit`，`flag_gems.device == "cpu"` |

厂商 wheel 缺少的共享库**没有装进系统**：用 `apt-get download` +
`dpkg -x` 解到 `$HOME/petvision-k1/libs/root`，只通过 `LD_LIBRARY_PATH` 生效：

| 库 | 版本 |
| --- | --- |
| `libsleef.so.3` | `libsleef3 3.6-1~exp1` |
| `libomp.so.5` | `libomp5-18 1:18.1.8-11bb4` |
| `libnuma.so.1` | `libnuma1 2.0.18-1build1` |

在 K1 上从本仓库根目录执行。共享库依赖 Bianbu 软件源提供的对应版本。

```bash
export PETVISION_K1="$HOME/petvision-k1"
mkdir -p "$PETVISION_K1/libs/root"
python3 -m venv --without-pip "$PETVISION_K1/venv"
k1_pip=$(mktemp)
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$k1_pip"
"$PETVISION_K1/venv/bin/python" "$k1_pip" -q
rm -f -- "$k1_pip"
. "$PETVISION_K1/venv/bin/activate"
python -m pip install --index-url https://git.spacemit.com/api/v4/projects/33/packages/pypi/simple \
  --extra-index-url https://pypi.org/simple \
  "torch==2.8.0+spacemit.0" "torchvision==0.23.0" "triton==3.6.0+spacemit.a5"
(
  cd "$PETVISION_K1/libs"
  apt-get download 'libsleef3=3.6-1~exp1' 'libomp5-18=1:18.1.8-11bb4' 'libnuma1=2.0.18-1build1'
  for deb in ./*.deb; do dpkg-deb -x "$deb" root; done
)
git clone https://github.com/flagos-ai/FlagGems.git "$PETVISION_K1/FlagGems"
git -C "$PETVISION_K1/FlagGems" checkout --detach "$(cat third_party/FLAGGEMS_COMMIT.txt)"
git -C "$PETVISION_K1/FlagGems" apply "$PWD/patches/0001-triton-libdevice-hypot-compat.patch" \
  "$PWD/patches/0002-tests-softmax-37-class-logits.patch"
python -m pip install 'packaging>=26.0' 'PyYAML==6.0.1' 'sqlalchemy==1.4.52' \
  'setuptools<77' 'setuptools-scm>=8,<10' 'wheel==0.46.2' pytest
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 python -m pip install --no-deps --no-build-isolation "$PETVISION_K1/FlagGems"
mkdir -p "$PETVISION_K1/app/scripts" "$PETVISION_K1/private"
cp petvision_forward.py flagos_env.py "$PETVISION_K1/app/"
cp scripts/k1_env.sh scripts/k1_ops_check.py scripts/k1_ab_model.py "$PETVISION_K1/app/scripts/"
```

**固定版导入兼容（仅隔离环境，不是上游已修复）**：SpacemiT 的
`triton.language.extra.cpu.libdevice`（a5 与 b1 都一样）只有 45 个符号，缺
`hypot`，而 FlagGems `_foreach_complex_math.py` 在导入期取
`tl_extra_shim.hypot`。部署树加了 `patches/0001-triton-libdevice-hypot-compat.patch`
（缺省回退到通用 `triton.language.extra.libdevice.hypot`），只影响未使用的复数
foreach 算子；这不是 PetVision 的性能/正确性结论。

## 3. 为什么选 `aten._softmax`

在 x86（torch 2.11）和 K1 上用 `TorchDispatchMode` 实测：

* `AdaptiveAvgPool2d(1)` 分解为 **1 次 `aten.mean.dim`**；
* ResNet-50 每前向 **53 次 `aten.convolution`**（53 个 Conv2d 前向钩子也计数为 53）；
* 37 类分类概率是 **1 次 `aten._softmax`**（`torch.softmax(logits, dim=1)`）。

固定版的 `mean.dim` 遇到第 5 节的故障，因此最初选择 `aten._softmax` 做真图验证。
注册名使用实现名 `softmax`（固定版 `only_enable` 的 include 语义），
`torch.softmax` 经 `aten._softmax` 派发到 `_spacemit.ops.softmax`。

## 4. 真机结果（原始 JSON 在 `results/k1/`）

### 4.1 完整 ResNet-50（同机、同权重）

| 输入 | 路径 | median | conv2d/前向 | max_abs_err vs x86 | Top-1 |
| --- | --- | --- | --- | --- | --- |
| seed 0，`randn(1,3,256,256)` | 原生 ATen | 2115.76 ms | 53 | `1.19e-06` | `leonberger` (p=0.050724) |
| `Abyssinian_201.jpg` | 原生 ATen | 2167.63 ms | 53 | `1.05e-05` | `Abyssinian` (p=0.997643857) |
| 同一张图 + 仅 `softmax` 注册（旧路径 artifact） | 前向仍是原生 ATen；后处理厂商 kernel | 2383.14 ms* | 53 | `1.05e-05` | `Abyssinian` (p=0.997643888) |

\* 上表最后一行的 2383.14 ms 来自旧路径 artifact，保留不改。当前
`--enable-ops softmax` 使用 **softmax-only 契约**：模型 logits 由原生路径产出，
不再执行/记录/重复计时 FlagGems 模型前向；只在概率计算时进入
`use_gems(include=["softmax"], record=False)`，模型时间取 `native_torch.timing`，
JSON 的 `flag_gems.model_path` 与 `flag_gems.timing.note` 会写明。
同一次交替 A/B 的结果见下表。真图→概率路径的厂商 kernel 计数为 **1**，
概率与原生差 `1.42e-14`。

计时：`time.perf_counter`，同进程、同 shape/dtype，`torch.set_num_threads(8)`，
CPU 亲和 0–7。seed 0 预热 3 次、测量 10 次。
真图: warmup 1 + repeat 5。派发 tracing 与计时分离。

**同一次交替 A/B（`scripts/k1_ab_model.py`，`results/k1/ab_image.json`）**：
同一模型/同一真图/8 线程，10 对，每对按奇偶交换 native/vendor 顺序，
`record=False`、无 hook、每对分别计时模型前向与 softmax：

| 指标 | native | vendor | 说明 |
| --- | --- | --- | --- |
| 模型前向 median | 2306.28 ms | 2255.56 ms | 两臂都使用原生 ATen 执行前向，无法将差值归因于 softmax |
| 模型前向加 softmax median | 2306.42 ms | 2257.36 ms | 对每轮两段耗时求和后取中位数 |
| 37 类 softmax median | 0.113 ms | 1.195 ms | 厂商 kernel 慢约 10.6×，不声称收益 |
| 概率 max_abs_diff | — | `1.42e-14` | Top-1 一致 |

`topk_summary*` 只展示 batch 的**第 0 张**（JSON 中 `sample: 0`）；`--batch > 1`
时 `logits_metrics` 的 `top1_*` 仍是旧的“flatten 后”语义，JSON 中写有
`batch_note`。`--image` 与 `--batch != 1` 会被明确拒绝。

### 4.2 单算子（`results/k1/ops_softmax.json`）

| 算子 | 原生 median | 厂商 median | 数值 | kernel launch |
| --- | --- | --- | --- | --- |
| `(1,37) FP32` softmax | 0.0353 ms | 0.5902 ms | max_abs `5.96e-08` | 1 |

厂商 softmax 慢约 16×（小 shape、libtuner 配置），**不声称性能收益**。

### 4.3 派发证据

* `impl.softmax_module == "_spacemit.ops.softmax"`；
* `probabilities.softmax_kernel_runs == 1`（`softmax_kernel_spacemit.run` 真实计数）；
* `TRITON_CACHE_DIR` 新增编译产物（JIT 执行）；
* `checks` 全为 true：`softmax_kernel_invoked_once`、`softmax_probs_match`、
  `softmax_top1_agree`、`reference_allclose`、`reference_top1_agree`。

## 5. 失败与未解决问题（不掩盖）

### 5.1 厂商 `conv2d`：拆分 barrier 后在 K1 通过

固定版原实现把 im2col 和 GEMM 融合在一个 kernel 中，并调用
`smt.global_mbarrier`。K1 的 Triton semantic 层直接报：

```
NotImplementedError: global_mbarrier Not supported: The syntax is not currently
implemented in the lowering.
```

把 wheel 中已有但被 Python 层挡住的 builder 路径恢复后，预编译
`spine-opt` 又报 `custom op 'xsmt.global_mbarrier_init' is unknown`；这条路径需要
重建 FlagTree 编译器。此外，原融合 kernel 的 im2col 列布局与
`weight.reshape(OC, -1)` 不一致，不能只取消 `raise`。

上游修复改为两个顺序 kernel：

1. `im2col_kernel` 按 `[channel][kernel_h][kernel_w]` 写入临时 buffer；
2. `bmm_kernel` 使用 K 分块和 `smt.dot` 累加，再一次性加 bias。

两个 kernel 都限制为最多 512 个程序并用 grid-stride loop，覆盖 K1 的 stem 大网格；
尾部列和 bias 使用掩码。当前显式支持 FP32、`groups=1`、整数或二元组
stride/padding/dilation；训练调用保留通用 FlagGems autograd 路径，其他推理参数
显式抛出 `NotImplementedError`。

K1 直接调用与 `use_gems(include=["conv2d"])` 注册调用均通过；4 个算子用例和
256×256 ResNet stem 全部 finite、`allclose(1e-3)`，stem 对 FP64 的最大绝对误差
为 `3.05e-05`。上游 `tests/test_conv2d.py -k spacemit` 为 **5 passed,
72 deselected**。

固定 PetVision 权重和 `Abyssinian_201.jpg` 的 53 次卷积全部使用
`im2col_kernel`/`bmm_kernel`（各 53 次），没有 native Conv2d fallback；Top-1
仍为 `Abyssinian`，相对 x86 logits 最大误差 `7.63e-06`，相对 K1 原生路径
`3.81e-06`。同机完整模型 median 为原生 `2097.0 ms`、厂商 Conv2d
`4135.6 ms`；stem 为 `129.7 ms`、`142.1 ms`，因此不声称性能收益。

[上游 PR #9823 与复现命令](https://github.com/flagos-ai/FlagGems/pull/9823)；
修复前的原始失败仍保留在 `results/k1/conv2d_vendor_failure.json`。

### 5.2 厂商 `mean.dim` 的固定版故障与上游修复

固定版原样（`(1,2048,8,8)`, `dim=[2,3]`, `keepdim=True`）在编译期失败：

```
AssertionError('Loop-carried variable _mean has initial type fp32 but is
re-assigned to fp64 in loop! Please make sure that the type stays consistent.')
```

`patches/0003-EXPERIMENTAL-spacemit-mean-dim-k1-does-not-fix.patch`
加入 `tl.sum(..., dtype=tl.float32)` 与 `padding_option="zero"` 后可以编译，
真实 launch 计数为 1，但输出为 **NaN**（`1×2048×8×8` 和 `1×2048×1×1` 都错），
与 fp64 参考 `allclose=false`。后续在 K1 上用独立的程序编号写出内核复现了调度问题。
512 个程序能写入全部输出，2048 个程序启动返回后，预填值全部保留。

上游补丁将单次启动限制在 512 个程序，并让每个程序按网格步长处理多行。归约使用
显式 FP32 向量累加和掩码加载，覆盖跨分块的 513 元素尾部。K1 真图前向调用一次厂商
均值内核。修复版单算子中位时间为 3.186 ms，同机 PyTorch 为 0.867 ms。
[上游 PR 与复现命令](https://github.com/flagos-ai/FlagGems/pull/9773)

### 5.3 其它

* 分割分支（U-Net / `ConvTranspose2d`）未移植、未运行；本报告只覆盖最终
  ResNet-50 分类路径。
* 未做 RVV/SMT 优化；没有可靠热点收益证据。

| `tests/test_softmax.py` | ✅ 可提交 | `patches/0002-tests-softmax-37-class-logits.patch`，确定性 FP32 `(1,37)` 前向用例。|
| SpacemiT `conv2d` | ✅ 可提交 | 两阶段 im2col+BMM 修复及回归测试，K1 真机和 PetVision 通过；[FlagGems PR #9823](https://github.com/flagos-ai/FlagGems/pull/9823)。|
| `patches/0001-triton-libdevice-hypot-compat.patch` | ⚠️ 环境兼容 | 仅为固定版在 SpacemiT Triton 上可导入；未证明其他后端需要 |
| `patches/0003-EXPERIMENTAL-spacemit-mean-dim-k1-does-not-fix.patch` | ❌ 不可提交 | 能过编译但结果 NaN，保留为失败复现 |

固定版的失败记录在 `results/k1/ops_mean_k1_failure.json` 和
`results/k1/conv2d_vendor_failure.json`。修复版的 mean.dim 回归用例位于
PR #9773，Conv2d 回归用例位于 PR #9823；PetVision 对照数据分别见
`results/k1/mean_dim_pr_9773.json` 和 `results/k1/conv2d_pr_9823.json`。

## 7. K1 部署（保留，供验收）

部署目录为 `$HOME/petvision-k1`。

| 路径 | 内容 | 保留 |
| --- | --- | --- |
| `venv/` | 已装 torch/torchvision/triton/FlagGems | ✅ |
| `app/` | `petvision_forward.py`、`flagos_env.py`、`scripts/` | ✅ |
| `FlagGems/` | 固定版源码（含 hypot 兼容） | ✅ |
| `libs/root/usr/lib/riscv64-linux-gnu/` | sleef/openmp/numa | ✅ |
| `private/best_slim.pt` | 94,647,883 B，320 个 state 项，`weights_only=True` 可读 | ✅ |
| `private/Abyssinian_201.jpg` | 验收图（46,667 B，不入 Git） | ✅ |
| `private/x86_seed0_logits.json`、`private/x86_image_logits.json` | x86 参考 | ✅ |
| `triton-cache/` | JIT 缓存 | ✅ |
| `results/k1/` | 板端原始 JSON | ✅ |

从空环境复现时，先在 x86 的课设 PyTorch 环境中设置 `PETVISION_ORIGINAL_WEIGHTS`
和 `PETVISION_IMAGE`，分别指向本人训练的 `raw/best.pt` 和测试图 `Abyssinian_201.jpg`。
在本仓库根目录生成部署文件：

```bash
python - "$PETVISION_ORIGINAL_WEIGHTS" <<'PY'
import sys
import torch

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
fields = ("model_state", "class_names", "architecture", "image_size", "mode", "precision", "epoch")
torch.save({key: checkpoint[key] for key in fields}, "best_slim.pt")
PY
python petvision_forward.py --backend torch --threads 1 --weights best_slim.pt \
  --image "$PETVISION_IMAGE" --save-logits x86_image_logits.json --out results/x86_reference
```

将 `best_slim.pt`、测试图和 `x86_image_logits.json` 放入板端 `private/`。固定种子输入的参考值使用
本仓库的 `results/local_x86_cpu_20261122/model/logits.json`。

验收命令（板端）：

```bash
export PETVISION_K1="$HOME/petvision-k1"
export PETVISION_REPO="/path/to/petvision-flagos-riscv"
cp "$PETVISION_REPO/scripts/k1_ops_check.py" "$PETVISION_K1/app/scripts/k1_ops_check.py"
cp "$PETVISION_REPO/petvision_forward.py" "$PETVISION_K1/app/petvision_forward.py"

# Conv2d PR branch: install the fixed source in the isolated K1 environment.
git -C "$PETVISION_K1/FlagGems" fetch https://github.com/LizhToGo/FlagGems.git \
  spacemit-conv2d-two-kernel
git -C "$PETVISION_K1/FlagGems" checkout --detach \
  5481b6a3e84d1a5a5bb4087cf9a9f28ae88d1335
git -C "$PETVISION_K1/FlagGems" apply \
  "$PETVISION_REPO/patches/0001-triton-libdevice-hypot-compat.patch"
SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 "$PETVISION_K1/venv/bin/python" -m pip install \
  --no-deps --no-build-isolation "$PETVISION_K1/FlagGems"

cd "$PETVISION_K1/app" && . scripts/k1_env.sh
python scripts/k1_ops_check.py --only conv2d --out /tmp/k1_conv2d_pr
cd "$PETVISION_K1/FlagGems" && python -m pytest tests/test_conv2d.py -k spacemit -q
cd "$PETVISION_K1/app"
python petvision_forward.py --backend flag_gems --enable-ops conv2d \
  --require-vendor spacemit --threads 8 --warmup 1 --repeat 3 \
  --weights ../private/best_slim.pt --image ../private/Abyssinian_201.jpg \
  --reference-logits ../private/x86_image_logits.json \
  --out /tmp/k1_petvision_conv2d_pr

# Softmax-only acceptance (keeps the original deployment path).
python scripts/k1_ops_check.py --only softmax --out /tmp/k1_accept_ops
python petvision_forward.py --backend torch --threads 8 --warmup 1 --repeat 5 \
  --weights ../private/best_slim.pt --image ../private/Abyssinian_201.jpg \
  --reference-logits ../private/x86_image_logits.json \
  --out /tmp/k1_accept_native
python petvision_forward.py --backend flag_gems --enable-ops softmax --threads 8 \
  --warmup 1 --repeat 5 --weights ../private/best_slim.pt \
  --image ../private/Abyssinian_201.jpg \
  --reference-logits ../private/x86_image_logits.json \
  --out /tmp/k1_accept_softmax
python scripts/k1_ab_model.py --threads 8 --pairs 10 --weights ../private/best_slim.pt \
  --image ../private/Abyssinian_201.jpg --out /tmp/k1_ab_image.json
cd "$PETVISION_K1/FlagGems" && python -m pytest tests/test_softmax.py -k 37_class_logits -q
```

## 8. 验证范围

本次运行覆盖 PetVision 的最终 ResNet-50 分类路径。原始测量位于 `results/k1/`，
源权重和测试图由使用者在私有目录中提供。