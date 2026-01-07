# results.md —— GRPO 对齐实验记录（TRL）

本文件用于记录：训练配置、训练前后多次采样输出、以及在 LLM-RL（RLHF）实验中观察到的典型现象（如 reward hacking / 退化生成）与对应修正思路。  
**目标**：让读者能快速看懂“我做了什么、发生了什么、我怎么分析与改进”。

---

## 1. 实验配置（Experiment Setup）

### 1.1 模型与框架
- 框架：Hugging Face TRL
- 算法：GRPO（Group Relative Policy Optimization）
- 基座模型： Qwen/Qwen2.5-0.5B-Instruct 
- 设备：Apple Silicon（训练阶段可能使用 MPS；生成评估阶段强制切回 CPU 以避免 device mismatch）

### 1.2 关键超参数（核心影响行为）
- `num_generations`（group size）：4
- `max_steps`：150
- `max_completion_length` / `max_new_tokens`：128
- 评估采样参数：
  - `do_sample=True`
  - `temperature`：0.7

---

## 2. Prompt 与 Reward 设计

### 2.1 Prompt
```text
请给我一个三点式学习计划，用 '-' 做 bullet，
每一条都必须包含一个时间（如 30 分钟），
不要有多余解释，只输出三条 bullet。

###2.2 Reward 设计（Rule-based）

Reward 主要由以下部分组成（可按你代码实际情况微调）：

结构约束：是否严格使用 - 开头的 bullet，且数量为 3 条

内容约束：是否包含时间表达（如“30分钟”“1小时”）

硬约束/反退化项（用于抑制常见退化行为）：

惩罚 prompt 复述

惩罚过长输出或重复 token（如 “API/API/...”）

备注：LLM 的 RL 对齐实验常出现 reward hacking / 退化策略（模型用无意义输出规避学习目标）。
在早期阶段，加入硬约束与更“尖锐”的 reward shaping 往往比“盲目加训练步数”更有效。

##3. 训练前后输出对比（Before vs After）
###3.1 训练前（Before Training）
请给我一个三点式学习计划，用 '-' 做 bullet，每一条都包含一个时间（如 30 分钟）。 - 第一步：设定目标 - 第二步：制定计划 - 第三步：实践行动 
当然可以！以下是一个基于您要求的三个步骤的学习计划： 1. **第一步：设定目标** - 目标设定是开始任何学习活动的基础。你可能想要设定短期或长期的目标，例如： - 学习新语言 - 约 60 分钟 - 提高数学成绩 - 约 45 分钟 - 开拓社交网络 - 约 30 分钟

###3.2 训练后（After Training，多次采样）

训练后不看单次输出，而看多次采样（例如 5 次）以观察生成分布是否发生偏移。
After Sample 1
请给我一个三点式学习计划，用 '-' 做 bullet,每一条都包含一个时间（如 30 分钟）,不要有多余解释，只输出三条bullet。当然，对阻阻阻阻阻阻阻...
After Sample 2
请给我一个三点式学习计划，用 '-' 做 bullet,每一条都包含一个时间（如 30 分钟）,不要有多余解释，只输出三条bullet。 -    1、 《关于社会主义建设、代码

给定/API/API/API/API/API/API/...

##4. 观察现象与诊断（Observations & Diagnosis）
###4.1 现象：reward hacking / 退化生成（degenerate policy）

表现：

模型倾向于复述 prompt 或生成大量无意义 token（例如 API/API/...）

输出不再追求满足 reward 中的“结构+时间”约束

解释（RL 视角）：

当 reward 信号稀疏/脆弱、或惩罚不够强时，策略可能找到一种“低方差、风险更小”的行为来规避学习目标

在 LLM 对齐中，这是非常典型的 early-stage failure mode：模型学会逃避 reward，而不是满足 reward

##5. 修正策略（Mitigations）

下面是本实验中可行的修正方向

【 】加强硬约束：对 prompt 复述、过长输出、重复 token 给强负分

【 】reward 更“尖锐”：对 “bullet != 3” / “缺少时间” 直接给明显负分

【 】降低采样温度：如从 0.7 降到 0.3，先稳定行为再增加步数

【 】扩大 prompt 集合：从 1 条 prompt 扩到 10–50 条，降低过拟合与退化风险

【 】在稳定后再加训练步数：仅加 steps 往往会放大退化

经验总结：

在 LLM-RL 对齐里，早期效果通常由 reward 设计 + 约束项 + 采样策略决定，而不是“训练更久”。

##6. 工程与复现要点（Engineering Notes）

pad_token：需要显式设置（常用做法是 pad_token = eos_token），否则 batch padding / generate 可能异常

device：训练用 MPS 时，生成评估阶段建议强制切回 CPU，避免 MPS placeholder / device mismatch 错误

常见 warning：

use_cache=True 与 gradient checkpointing 不兼容（框架会自动关闭 cache）

tokenizer token id 对齐提示（一般不影响训练，可记录）

###7. 下一步计划（Next Steps）

 完成一版“硬约束 + 低温采样”的 reward shaping，并记录修正前后对比

 增加 prompt 集合并统计合规率（pass rate）

 形成更完整的对照：SFT-only vs GRPO / Instruct vs Base


