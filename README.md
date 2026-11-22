# PetVision × FlagOS（SpacemiT 后端）迁移验证

把结课项目 **PetVision**（Oxford-IIIT Pet：U-Net 前景分割 + ResNet-50 品种分类）
迁移到 RISC-V 软件栈的**可复现验证工程**。原课设仓库
`C:/Users/27978/Desktop/认知科学与类脑计算` 只读使用：不复制其代码、不写入其目录、
不改动其 Git 状态，也不把权重提交进本仓库。

目标后端：

| 项目 | 版本 / 来源 |
| --- | --- |
| 算子库 | FlagGems，commit `43c9cab402f7e286de9cbfc6238ea3e7bfe354cb`（`third_party/FLAGGEMS_COMMIT.txt`） |
| 编译器 | FlagTree / Spine-Triton **3.6**（`triton==3.6.0+spacemit.a5`） |
| 厂商后端 | FlagGems `runtime/backend/_spacemit`（vendor 名 `spacemit`，`GEMS_VENDOR=spacemit`） |
| 目标硬件 | SpacemiT K3（RISC-V 64，RVV vlen=1024 / elen=64） |
| 开发环境 | SpacemiT 官方 **x64 emulation** 镜像 `harbor.baai.ac.cn/flagtree/flagtree-spacemit-py312-torch2.10.0-ubuntu24.04:202607-3.6-base`（manifest config digest `sha256:49afe94e8aa50161ab8176b3339169e283654b4064db7ed3c1adfe0d8c2c44b8`） |

## 1. 结论摘要（先看这里）

1. **算子选择有运行证据**：ResNet-50 分类模型每个前向实际调用 `aten::conv2d`
   **53 次**（`torch.nn.modules.conv.Conv2d` 前向钩子计数，见
   `results/local_x86_cpu_20261122/model/petvision_forward.json`），权重里 53 个 4 维
   卷积核全部 `groups=1`；U-Net 同样以 `Conv2d(3,3,padding=1)` 为基本单元。因此
   Conv2d 是 PetVision 上最有适配价值的候选算子，而不是任意挑选。
2. **不要盲目取消注释**：`_spacemit/ops/conv2d.py` 从 PR #2527 引入起就存在，但
   `_spacemit/ops/__init__.py` 里 `from .conv2d import conv2d` 一直是注释状态（同一
   PR 的 `conv1d` / `thnn_conv2d` / `_conv_depthwise2d` 也是注释）。后果是
   `aten::conv2d` 在 SpacemiT 上落到**通用 CUDA 取向实现**，而不是厂商融合
   im2col+GEMM kernel。原因与证据链见第 6 节。
3. **实现确有缺陷**（修复前）：kernel 完全忽略 `groups` 参数（`_conv_depthwise2d`
   包装器却按分组语义调用它，会读到越界/错位权重），并且不接受 `"same"/"valid"`
   字符串 padding（测试直接调用 `flag_gems.conv2d(padding="same")` 会得到
   `ValueError`）。这些是"静默算错"级别的根因，必须先显式拒绝再导出。
4. **交付的上游候选补丁**：`patches/0001-spacemit-conv2d-enable-forward-kernel.patch`
   与 `patches/0002-tests-spacemit-conv2d.patch`。前者导出 conv2d + 显式限制 +
   测试适配（在 `tests/test_conv2d.py` 内以 `vendor_name == "spacemit"` 跳过梯度段，
   因为该 kernel 是推理前向实现、没有 autograd），后者是 SpacemiT 专属注册/限制测试。
5. **证据分级**（不得混用）：

| 环境 | 数值正确性 | 性能 | 状态 |
| --- | --- | --- | --- |
| 本地 x86_64 CPU + PyTorch 2.11（Windows） | ✅ 5 个 conv2d case + 真实 ResNet-50 前向 | ✅ 有 JSON | 已验证（本仓库 `results/local_x86_cpu_20261122/`） |
| SpacemiT x64 emulation（x86 主机 + RISC-V 用户态仿真） | ❌ 未执行 | ❌ 未采集 | 镜像已解包且 chroot 可运行，但镜像缺 triton/flagtree/RPC runtime；证据见 `results/spacemit_emu_20261122/environment_attempt.json` |
| SpacemiT K3 真机 | — | — | **未取得真机证据**，本仓库不声称任何 RVV 加速比 |

## 2. 目录结构

```
conv2d_case.py                 最小 conv2d 正确性 + 计时（参考实现：fp64 CPU torch）
petvision_forward.py           真实 ResNet-50 权重前向 + 计时 + conv2d 调用计数
flagos_env.py                  版本/环境采集、warmup+repeat 计时、误差指标
tools/fetch_oci_rootfs.py      无 Docker 守护进程地把镜像流式解包成 rootfs（本次远端方案）
patches/                       上游候选补丁（对 FlagGems 固定 commit 生成，git apply 可用）
results/local_x86_cpu_20261122/  本地 CPU 原始 JSON（真实执行结果）
results/spacemit_emu_20261122/   仿真环境原始 JSON/日志（若执行成功）
third_party/FLAGGEMS_COMMIT.txt  上游 commit 固定点
```

`third_party/FlagGems/` 是**不入库**的只读检出目录（`.gitignore` 排除），用于生成/校验补丁：

```bash
git clone https://github.com/FlagOpen/FlagGems.git third_party/FlagGems
git -C third_party/FlagGems checkout 43c9cab402f7e286de9cbfc6238ea3e7bfe354cb
```

## 3. 从原课设取权重（不复制、不写入）

本工程**不 import 课程包**，只按 checkpoint 的键名复刻同样的结构：

* `backbone.*` → `torchvision.models.resnet50(weights=None)`（`fc` 换成 `Identity`）
* `classifier.0.*` → `nn.Linear(2048, 37)`
* `load_state_dict(..., strict=True)`，失败即报错

权重通过显式参数传入，默认只读打开：

```bash
--weights C:/Users/27978/Downloads/111/resnet50-256-color-raw-20261103-212216-24717/raw/best.pt
```

脚本只记录 `size_bytes` 与 `mtime_ns`（运行前后对比），并在 JSON 中写
`"unchanged": true`；不做整文件哈希。仓库内没有任何 `.pt`/`.pth`（`.gitignore` 已排除）。分割权重
`segmentation-320-.../best.pt` 没有用于本次运行，原因见第 8 节。

## 4. 运行前提

* Python ≥ 3.10，`torch` + `torchvision`（本地 CPU 即可；本仓库在
  `torch 2.11.0+cu128` / `torchvision 0.26.0` 上验证，CPU 后端）。
* `conv2d_case.py` 不需要 FlagGems，也不需要任何私有路径，可直接复现参考数值与耗时。
* `--backend flag_gems` 需要 FlagGems + 目标后端（`GEMS_VENDOR=spacemit` + SpacemiT
  Triton 插件）。本地 Windows 没有 SpacemiT Triton，因此本地只跑 `--backend torch`。

最小本地窄检查（任何有 torch 的环境；本次执行过，日志在
`results/local_x86_cpu_20261122/*/run.log`）：

```bash
python -m py_compile conv2d_case.py petvision_forward.py flagos_env.py \
    tools/fetch_oci_rootfs.py
python conv2d_case.py --backend torch --out results/local_check/conv2d
python petvision_forward.py --backend torch --weights <best.pt> \
    --save-logits results/local_check/logits.json --out results/local_check/model
```

补丁自身的最小校验（不需要 GPU/SpacemiT 硬件）：

```bash
cd third_party/FlagGems
git apply --check ../../patches/0001-spacemit-conv2d-enable-forward-kernel.patch
git apply --check ../../patches/0002-tests-spacemit-conv2d.patch
```

## 5. SpacemiT 环境（官方镜像）

官方文档给出的两条路（`harbor.baai.ac.cn` 可访问，实测 manifest 总计 **11.47 GB
压缩 / 19.4 GB 解包**）：

```bash
IMAGE=harbor.baai.ac.cn/flagtree/flagtree-spacemit-py312-torch2.10.0-ubuntu24.04:202607-3.6-base
docker pull "$IMAGE"          # 文档标注 19.4GB
# 或
wget https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-spacemit-py312-torch2.10.0-ubuntu24.04.202607-3.6-base.tar.gz
docker load -i flagtree-spacemit-py312-torch2.10.0-ubuntu24.04.202607-3.6-base.tar.gz
```

容器内：`triton==3.6.0+spacemit.a5`，x86 侧做编译、RISC-V 侧由
`spine-rpc-server`（jdsk-qemu 用户态仿真，`SPINE_TRITON_TARGET_ARCH=riscv64`）执行
kernel。运行算子前需：

```bash
export GEMS_VENDOR=spacemit
export SPINE_TRITON_TARGET_ARCH=riscv64
```

（FlagTree CI 用 self-hosted runner `spacemit3.6` + jdsk-qemu v10.0.2
`-cpu max,vlen=1024,elen=64,vext_spec=v1.0` 启动 spine-rpc-server，经 TCP 9999 下发
kernel；该工作流文件在 `flagos-ai/FlagTree` 仓库内，非本仓库资产。）

### 5.1 无 Docker 的 rootfs 方案（本次使用的替代路径）

本次可用的 x86 主机只有 2 vCPU / 2 GiB RAM / 30 GB 空闲磁盘，`docker pull` 会同时保留
压缩 blob 与解包快照（约 31 GB），因此改用流式解包到普通目录：

```bash
sudo python3 tools/fetch_oci_rootfs.py \
    --repo flagtree/flagtree-spacemit-py312-torch2.10.0-ubuntu24.04 \
    --tag 202607-3.6-base --dest /var/tmp/sm-rootfs
```

然后把本仓库拷进 rootfs 并在 chroot 中运行（`/proc`、`/sys`、`/dev` 需 bind mount），
完整命令见 `scripts/run_in_spacemit_rootfs.sh`。本次实测的资源事实：

| 事实 | 数值 |
| --- | --- |
| 镜像 manifest 压缩总量 | 11.47 GB（11 层，最大层 8.86 GB） |
| 官方文档标注解包后体积 / `docker pull` 体积 | 11 GB 归档 / 19.4 GB 镜像 |
| 远端主机 | 2 vCPU，1962 MiB RAM + 2048 MiB swap，`/` 空闲 30 GB，`/tmp` 是 982 MiB tmpfs |
| Harbor 可访问性 | 可访问；实测带宽 ≈ 13 MiB/s |
| GitHub Releases | 30 s 内 0 字节（`curl` 超时），未能下载 RISC-V runtime 资产 |

结论：`docker pull` 需要"压缩 blob + 解包快照"两份存储，超出空闲磁盘，故改用流式
rootfs。本次实际做到"解包成功 + chroot 可运行"，但镜像内缺 triton/flagtree/RPC
runtime，因此 **SpacemiT kernel 未执行**；失败事实完整记录在
`results/spacemit_emu_20261122/environment_attempt.json`，不得用本地 CPU 结果替代。

## 6. 为什么 conv2d 之前没有被导出（注册链证据）

1. `flag_gems/__init__.py` 先执行
   `SpecOpRegistrar(registry=globals(), vendor=vendor_name).apply()`（第 66–67 行），
   再用这些全局名字构造 `_FULL_CONFIG`（第 94 行起）。因此 backend 的 `ops/__init__.py`
   决定 `("conv2d", conv2d)` 指向**厂商实现还是通用实现**。
2. `_spacemit/ops/__init__.py` 把 `from .conv2d import conv2d` 注释掉后，
   `import_vendor_extra_lib` 里 `inspect.getmembers(ops_module, isfunction)` 拿不到
   `conv2d`，`SpecOpRegistrar` 不会覆盖，`aten::conv2d` 注册的是
   `flag_gems.ops.conv2d`（CUDA 取向，实现里假设 GPU/`runtime` 语义）。
3. 该注释从 PR #2527（"Add Spacemit runtime backend with operator implementations"）
   引入时就存在，且同一 PR 把 conv 测试在 spacemit 上 skip（`git log -S` 可复现）。
   现在 CI 配置 `.github/backends.json` 里 spacemit `enabled: true`，`ops-test` 用
   `tools/run_tests.py --stages all` 全量跑测试；`tests/test_conv2d.py` 含反向传播段，
   而该 kernel 是纯前向函数（没有 `torch.autograd.Function`）。所以"只取消注释"
   必然在 spacemit CI 上失败——这就是必须先做接口限制 + 测试适配的原因。
4. 其它缺陷（本次修复）：`groups` 被忽略（`conv1d`/`thnn_conv2d`/
   `_conv_depthwise2d` 都以 `groups` 语义调用它）；字符串 padding 不支持；导出前
   没有任何 backend 级测试。

### 6.1 接口/形状限制（补丁后）

| 参数 | 支持情况 | 处理 |
| --- | --- | --- |
| `groups=1` | ✅ | 正常走融合 im2col+GEMM |
| `groups>1`（含 depthwise） | ❌ | 抛 `NotImplementedError`（修复前：静默错误） |
| `padding=int/tuple` | ✅ | 支持 |
| `padding="same"/"valid"` | ❌ | 抛 `NotImplementedError`（即使 `"valid"` 也显式拒绝，避免语义分叉） |
| `dilation>1` | ✅ | kernel 内已实现，本次用例含 `dilation=2` |
| `bias` 有/无 | ✅ | `HAS_BIAS` 常量分支 |
| dtype | fp32（本仓库验证） | fp16 未在目标环境验证，不声称支持 |
| 反向传播 | ❌ | 无 autograd 实现；`tests/test_conv2d.py` 在 spacemit 上跳过梯度段 |

## 7. 数值与性能结果

只记录**真实执行**的结果；同一 JSON 内不会把不同机器拼表。

### 7.1 本地 x86_64 CPU（PyTorch 2.11，Windows 11）

参考实现是 fp64 CPU 卷积，候选是原生 `aten::conv2d`（`--backend torch`，
`--device auto` 解析为 `cpu`），`warmup=5, repeat=20`，同步策略：CPU eager，
无需 synchronize（JSON 中 `config.device`/`timing.sync` 记录了实际值）。

| case | input | weight | stride/pad/dil | bias | max_abs_err | median |
| --- | --- | --- | --- | --- | --- | --- |
| unet_conv3x3_pad1_bias | 1×3×32×32 | 8×3×3×3 | 1/1/1 | ✅ | 5.72e-06 | 0.033 ms |
| stride2_pad1_bias_odd_input | 1×4×17×15 | 6×4×3×3 | 2/1/1 | ✅ | 1.91e-06 | 0.026 ms |
| valid_nopad_nobias_batch2 | 2×3×19×19 | 5×3×3×3 | 1/0/1 | ❌ | 3.81e-06 | 0.140 ms |
| dilation2_pad2_bias | 1×3×15×15 | 4×3×3×3 | 1/2/2 | ✅ | 2.86e-06 | 0.052 ms |
| petvision_resnet50_stem_7x7_s2_pad3 | 1×3×256×256 | 64×3×7×7 | 2/3/1 | ❌ | 3.05e-05 | 0.659 ms |

真实模型（ResNet-50，256×256，batch=1，`warmup=1, repeat=3`）：
**47.88 ms / 前向（median）**，53 次 conv2d 调用/前向，权重 `unchanged: true`。
原始数据：`results/local_x86_cpu_20261122/`。

### 7.2 SpacemiT x64 emulation

**未取得执行证据**。本次单次尝试的完整事实记录在
`results/spacemit_emu_20261122/environment_attempt.json` 与
`results/spacemit_emu_20261122/fetch_image_rootfs.log`：

1. 镜像 11 个层、压缩 11.47 GB，用 `tools/fetch_oci_rootfs.py` 流式解包成功
   （18 GB rootfs，907 s，剩余磁盘 12.6 GB），未创建任何 docker 镜像/容器。
2. `chroot` 可运行：Python 3.12.3、`torch 2.10.0+cu128`、`torchvision 0.25.0`、
   `numpy 2.4.2`，工具链在 `/opt/spacemit/spacemit-toolchain-linux-glibc-x86_64-ame-20260612`。
3. **但镜像内没有 `triton`、没有 `flagtree`、没有 `~/.flagtree/spacemit`、
   没有 `spine-rpc-server`**（`pip list` 与全盘 `find` 均确认）。该 tag 是
   "build-from-source base"，不是可直接运行的 SpacemiT 后端；要装插件需 FlagTree 源码
   编译 + GitHub Releases 的 RISC-V runtime 资产，而本机 GitHub Releases 下载 30 s 收到
   0 字节（`git ls-remote` 可用），且 1962 MiB RAM / 2 vCPU 不足以构建 LLVM/Triton。
4. 因此 **没有 SpacemiT 算子执行、没有 RVV 数据**；本地 CPU 数字不得替代。尝试后已
   卸载 bind mount 并删除 18 GB rootfs 与临时目录，主机磁盘恢复到 30 GB 空闲，
   未触碰主机上已有的容器/镜像。

真机证据：**未取得**。

## 8. 范围、阻塞与下一步

* **分割模型未跑**：U-Net 上采样使用 `ConvTranspose2d`，SpacemiT 后端没有
  `conv_transpose2d` 实现；即使 conv2d 适配完成，U-Net 也无法在 FlagGems 下完整前向。
  这是 SpacemiT 后端缺口，不是本补丁试图掩盖的问题。
* **SpacemiT 执行未完成（最重要阻塞）**：官方 `202607-3.6-base` 镜像缺
  `triton`/`flagtree` 与 RISC-V `spine-rpc-server`，无法在本机仿真环境跑 kernel；
  详细环境事实见 `results/spacemit_emu_20261122/environment_attempt.json`。
* **未做 RVV 优化**：没有真机热点数据，任何"RVV 收益"都不会被声称。补丁只做正确性、
  注册与测试。
* **下一步**：① 在 FlagTree spacemit3.6 runner 上跑 `tests/test_conv2d.py` 全参数
  组合，确认 fp32 全形状通过、fp16 是否可用；② 若需要 depthwise/depthwise-like 模型，
  给 kernel 增加分组支持后再导出 `conv1d`/`thnn_conv2d`/`_conv_depthwise2d`；
  ③ K3 真机上对 PetVision 前向做 warmup+repeat 计时与 profile，再决定 RVV 优化点。

## 9. 上游候选补丁

```bash
cd third_party/FlagGems           # 固定 commit 43c9cab...
git apply ../../patches/0001-spacemit-conv2d-enable-forward-kernel.patch
git apply ../../patches/0002-tests-spacemit-conv2d.patch
python -m py_compile \
  src/flag_gems/runtime/backend/_spacemit/ops/conv2d.py \
  src/flag_gems/runtime/backend/_spacemit/ops/__init__.py \
  tests/test_conv2d.py tests/accuracy_utils.py tests/test_conv2d_spacemit.py
python tools/ci_checks/check_init_exports.py     # 29 个 backend ops __init__ 全部通过
```

补丁内容与理由：

| 文件 | 改动 | 理由 |
| --- | --- | --- |
| `_spacemit/ops/conv2d.py` | 增加 `groups != 1`、字符串 padding 的 `NotImplementedError` 与 docstring | 修复静默错误；限制可审查 |
| `_spacemit/ops/__init__.py` | 导出 `conv2d`（只导出它，`conv1d`/`thnn`/depthwise 仍注释） | 让 `aten::conv2d` 真正走厂商 kernel |
| `tests/test_conv2d.py` | 前向断言后 `if vendor_name == "spacemit": return`；`test_conv2d_padding` 在 spacemit 跳过 | 该 kernel 无 autograd；不支持字符串 padding。只改本文件，不动通用 `accuracy_utils.TestForwardOnly()`，避免扩大其他测试的语义 |
| `tests/test_conv2d_spacemit.py`（新增） | 断言注册到 `GEMS_SPACEMIT`；断言不支持参数抛错 | 防止"注释掉导出"回归 |

补丁对 `43c9cab4` 用 `git apply --check` 验证通过；`check_init_exports.py` 全通过；
所有改动文件 `py_compile` 通过。**未提交上游 PR、未 push 任何远端**。

## 10. 性能方法学

* 计时为同一进程、同一环境：`time.perf_counter`，先 `warmup` 次再 `repeat` 次，
  报告逐次样本、`median_ms`、`p50_ms`（与 median 等价，取中位数）、`mean/min/max/stdev`。
* 候选实现在**活动设备**上运行（`--backend flag_gems` 时 `--device auto` 取
  `flag_gems.device`，SpacemiT 后端为 `cpu`；GPU 后端则为对应设备），计时前后调用该设备
  的 synchronize；参考值始终在 CPU 上以 fp64 计算，比较前 `.cpu()`，JSON 中记
  `"sync": "none (eager CPU execution)"` 或 `torch.<dev>.synchronize()`。
* 参考实现与候选实现使用同一输入张量（固定 seed）、同一 dtype、同一 shape。
* 跨机器结果不合并：本地 CPU 表与仿真/真机结果分目录、分 JSON。
* 仿真环境里的"耗时"包含 qemu 用户态模拟与 RPC 往返，只能用于"是否跑通"，不能当 K3 性能。