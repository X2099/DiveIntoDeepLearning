# -*- coding: utf-8 -*-
"""
@File    : LoRA微调模型.py
@Time    : 2025/7/1 14:37
@Desc    : 
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

from utils import chat_data, chat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载预训练模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 2. 添加对话任务中常用的特殊token（用户、机器人标记、BOS、EOS等）
special_tokens = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "additional_special_tokens": ["<user>", "</user>", "<bot>", "</bot>"]
}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token  # GPT2 没有原生pad，使用eos占位
model.resize_token_embeddings(len(tokenizer))  # 重新扩展嵌入矩阵

# 3. 配置LoRA（低秩适配器）参数
lora_config = LoraConfig(
    r=8,  # 降维的秩，控制LoRA插入模块的参数规模
    lora_alpha=32,  # LoRA缩放因子
    target_modules=["c_attn", "c_proj"],  # GPT2中注意力的关键线性层名称
    lora_dropout=0.1,  # dropout，防止过拟合
    bias="none",  # 不训练bias参数
    task_type=TaskType.CAUSAL_LM  # 表明是自回归语言建模任务
)

# 将LoRA结构注入到原模型中，仅训练adapter部分参数
model = get_peft_model(model, lora_config)

# 4. 加载聊天语料，每行为一组完整的用户-助手对话
# 将 chat_data 转换为 Hugging Face Dataset 格式
dataset = Dataset.from_list(chat_data)


# 5. 将对话文本转换为模型输入格式（带标签input_ids）
def tokenize(example):
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding


# 批处理分词 + 去除原始字段
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# 6. 创建数据整理器（自动padding、生成labels等）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT是自回归模型，不使用掩码语言建模
)

# 7. 设置训练参数
training_args = TrainingArguments(
    output_dir="./chatgpt2-lora",  # 输出路径
    per_device_train_batch_size=4,  # 单卡batch size
    num_train_epochs=3,  # 训练轮次
    save_steps=500,  # 每500步保存一次模型
    save_total_limit=2,  # 最多保留2个checkpoint
    logging_steps=50,  # 每50步打印日志
    report_to="none",  # 不使用wandb等日志平台
    fp16=True  # 使用半精度训练以节省显存
)

# 8. 启动Trainer进行LoRA训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

# 9. 保存LoRA适配器权重（不是完整模型）和tokenizer
model.save_pretrained("./chatgpt2-lora")
tokenizer.save_pretrained("./chatgpt2-lora")
