# ASMR Spatial Lab

[English](README.md) | 简体中文

## 项目状态

截至 2026 年 9 月 27 日，项目暂时停止开发，保留仓库用于记录实验。作者试听更偏好 RTF，而非新增的 HRTF 和 Meta 基线，但整体效果仍未达到最初目标。这是主观试听反馈，不是通用性能结论；目前没有继续开发计划。

**新增：** 实测近场 HRTF 和 Meta 官方 3-block 神经双耳模型。使用 `run --all-methods` 或 `compare --all-methods` 生成五种同源中文配音对照。[安装、使用及限制说明](SPATIAL_COMPARISON.md)。位置是估计值，不是真实追踪数据；已跑通不代表听感一定改善。

一个用于记录实验的命令行测试项目：把日语双声道音声转换为中文配音，尝试同时保留说话人的音色和原录音的空间听感。

**目前主观效果一般。** 项目用于验证方法和保留实验过程，不是成熟配音产品，也不代表最佳空间音频方案。算法指标改善不等于听起来更像原作。

## 测试什么

输入是单人说话的日语双声道录音，可能包含左耳、右耳、移动和近远变化。实验主要回答：

- 克隆音色并翻译成中文后，能否保留原录音的双耳空间线索？
- 相比居中配音，空间处理能带来多大改善？
- 更细的频谱和相位迁移，是否比简单音量差/时间差迁移更自然？

输入不需要真实三维坐标或头部追踪数据。测试不处理多人分离或复杂混合音效，输出仅包含中文配音，不混入原日语来制造空间感。

## 怎么做

```text
日语双声道录音
  → Parakeet CTC 1.1B 识别与分句
  → DeepSeek V4 Flash 翻译
  → IndexTTS-2.0 音色克隆与逐句风格参考
  → Rubber Band 保音高时长对齐
  → 空间渲染：居中对照 / 双耳线索迁移 / 复数 RTF 迁移
```

音频在本地处理，翻译阶段仅向 DeepSeek 发送文本。三个空间版本复用同一份中文配音，避免把 TTS 随机变化误当成空间算法差异。

当前后端为 IndexTTS 2.0，早期实验记录使用过 2.5。缓存键区分版本：继续旧任务时会用 2.0 重新生成中文，不会误用 2.5 缓存；已有缓存仍保留。2.0 适配器不传入 2.5 专属的语言或原生时长参数，时长调整使用 Rubber Band。`compare` 根据已完成任务报告中的版本复用缓存。

### 基线一：双耳线索迁移

实现：[`spatial.py`](spatial.py)。

- 使用 GCC-PHAT 互相关估计双耳时间差（ITD），进行插值、置信度筛选和平滑。
- 在 28 个频带估计左右耳电平差（ILD），保留频率相关的左右能量分布。
- 用缓慢变化的响度和轻度相对频谱变化近似近远听感。
- 通过分帧频域增益、相位延迟和重叠相加，把这些线索迁移到中文。

### 基线二：时变复数相对传递函数（RTF）迁移

实现：[`spatial_rtf.py`](spatial_rtf.py)。

- 消除整体 ITD 对跨帧平均的影响，估计左右声道的局部空间协方差。
- 使用主特征向量估计复数相对传递函数，保留更细的差分频谱与频率相关相位。
- 根据相干性和能量判断可信程度；低置信度频点回退到基线一的线索。

两种方法都是 **DSP 信号处理**，不是预训练的神经双耳模型；也不是完整 HRTF、房间脉冲响应或真实距离的重建。

### 对照方式

`compare` 对已有中文缓存分别渲染 RTF、旧版和居中版，按整轨双声道联合 RMS 匹配电平。不会独立归一化左右耳，也不会重新调用识别、翻译或 TTS。这是电平匹配，不是精确的感知响度匹配。

## 已做的实验

- 短日语片段已跑通识别、翻译、克隆和空间渲染。
- 已构造人工左右移动的语音片段进行集成测试。
- 已在一段约 4 分 24 秒、52 句的音声上生成三版对照。
- 已用不同随机源信号和已知 FIR 滤波器测试差分频谱、相位恢复；复杂差分滤波条件下，RTF 的客观误差低于旧版。

这些结果只能说明管线可运行、部分线索可以迁移，**不能证明贴耳感、表演和空间氛围已接近原作**。真实素材上的低置信度频点仍较多，尚未建立系统的人类听评结果。

详细过程见 [VALIDATION.md](VALIDATION.md) 和 [RTF_VALIDATION.md](RTF_VALIDATION.md)。其中提及的音频、缓存和输出路径均为本地实验记录，不随公开仓库发布。

## 运行方式

这是依赖已有环境的 Windows 测试程序，**不是下载后即可独立运行的整合包**。

需要在相邻的 `asmr-next` 目录中准备好 [ASMR Dubber](https://github.com/EveningStudy/asmr-dubber) 源码、主 Python 环境、Parakeet 1.1B/CrispASR、IndexTTS-2.0 隔离环境及模型。适配器复用其 ASR 时间戳解析、翻译代码和本地资源。

```text
Projects/
├─ asmr-next/
└─ asmr-spatial-lab/
```

DeepSeek 密钥从进程环境变量 `DEEPSEEK_API_KEY` 或原项目本地配置读取，不复制到本项目。翻译调用会产生相应 API 费用。

```powershell
cd asmr-spatial-lab
.\run.ps1 doctor
.\run.ps1 run "D:\audio\sample.wav"
```

建议先使用 15～60 秒的双声道片段，单次最多 10 分钟。单声道可加 `--allow-mono` 测试配音，但无法恢复原本不存在的双耳信息。

```powershell
# 翻译后停止，编辑 transcript.json 的 zh 字段，再继续
.\run.ps1 run "D:\audio\sample.wav" --out "outputs\test" --stop-after translate
.\run.ps1 resume "outputs\test"

# 对已有任务生成电平匹配对照，结果写入新目录
.\run.ps1 compare "outputs\test"
```

普通任务输出 `zh_spatial.wav`、`zh_rtf.wav`、`zh_center.wav`、中文字幕和诊断记录。`compare` 输出文件名为 `zh_v1.wav`、`zh_rtf.wav`、`zh_center.wav`。音频格式为 48 kHz、24-bit 双声道；这不意味着恢复了原录音全部高频细节。

如果另用 Python，可安装 `requirements.txt` 后执行 `python dub.py ...`，但仍需上述 ASMR Dubber 资源；非默认位置可用 `--asmr-root` 指定。

```powershell
..\asmr-next\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\asmr-next\.venv\Scripts\python.exe benchmark_rtf.py
```

## 限制

- 位置像，不等于气声、音色和表演像；空间处理无法补救 TTS 本身的差异。
- 近远只是响度/频谱近似，不能可靠区分靠近与说话变大声。
- 不恢复厘米级距离、前后/上下位置、完整房间混响或个体耳廓响应。
- 中文按句子时间窗放置，空间轨迹按句内进度映射，不保证词语、动作和呼吸逐点对应。
- 不做源分离，识别句子之外输出静音，不自动保留原呼吸和音效。
- 强混响、多人、耳间独立信号、遮挡和快速移动可能导致估计不可靠或音色失真。
- 没有实现 BinauralGrad、BinauralFlow；实测近场 HRTF 和 Meta BinauralSpeechSynthesis 已作为可选对照后端接入，见上方独立说明。

公开内容仅为源码、测试和实验说明，不包含原音频、生成音频、模型权重、密钥或运行缓存。请使用具有相应权限的素材，不将克隆结果冒充真实本人录音。
