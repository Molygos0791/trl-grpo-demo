# train_grpo.py
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

# -------------------------
# 1) 准备训练数据：只有 prompt
# -------------------------
prompts = [
    "请给我一个三点式学习计划，用 '-' 做 bullet,每一条都包含一个时间（如 30 分钟）,不要有多余解释，只输出三条bullet。",
]

train_dataset = Dataset.from_dict({
    "prompt": prompts
})

def generate_one(model_name,prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# 2) 定义 reward 函数
# -------------------------
def reward_fn(prompts, completions, **kwargs):
    rewards = []

    for prompt, completion in zip(prompts, completions):
        score = 0.0
        text = completion.strip()

        # 规则 1：统计 '-' bullet 数量
        bullets = len([
            line for line in text.splitlines()
            if line.strip().startswith("-")
        ])
        score += min(bullets, 3)

        # 规则 2：是否包含时间
        if re.search(r"\b\d+\s*(分钟|小时)\b", text):
            score += 1.0
        # 规则 3：惩罚项：
        if bullets ==0:
            score -= 3.0
        if bullets !=3:
            score -=1.0
        if not re.search(r"\b\d+\s*(分钟|小时)\b", text):
            score -= 2.0
        rewards.append(score)

    return rewards

# -------------------------
# 3) GRPO 训练配置
# -------------------------
config = GRPOConfig(
    output_dir="outputs",
    per_device_train_batch_size=4,

    # 关键参数：同一个 prompt 生成几条回答
    num_generations=4,

    # 先跑很少的 step，只验证流程
    max_steps=150,
    logging_steps=1,

    max_completion_length=128,
)
#####
print("\n===== BEFORE TRAINING =====")
before_text = generate_one(
    "Qwen/Qwen2.5-0.5B-Instruct", 
    prompts[0]
)
print(before_text)

# -------------------------
# 4) 创建 GRPOTrainer
# -------------------------
trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_fn,
    train_dataset=train_dataset,
    args=config,
)

# -------------------------
# 5) 开始训练
# -------------------------
trainer.train()
print("\n===== AFTER TRAINING (MULTIPLE SAMPLES) =====")

# 强制把模型移回 CPU（避免 MPS 冲突）
trainer.model = trainer.model.to("cpu")

for i in range(5):
    inputs = trainer.tokenizer(prompts[0], return_tensors="pt")

    with torch.no_grad():
        out = trainer.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )

    text = trainer.tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"\nSample {i + 1}:")
    print(text)

print("\nGRPO demo finished.")

