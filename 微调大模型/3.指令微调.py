# -*- coding: utf-8 -*-
"""
@File    : 3.指令微调.py
@Time    : 2025/7/1 17:10
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
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

from utils import chat_instruction

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 加载模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# 3. 添加特殊 token
special_tokens = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>"
}
tokenizer.add_special_tokens(special_tokens)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# 4. 配置 LoRA 参数
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config).to(device)

# 5. 准备指令微调数据
chat_data = [
    {
        "instruction": "介绍你自己",
        "input": "",
        "output": "我是一个由 GPT 模型训练的中文智能助手，可以回答问题、写作、翻译和聊天等。"
    },
    {
        "instruction": "写一首关于春天的诗",
        "input": "",
        "output": "春风拂面百花开，柳绿莺啼燕归来。暖阳轻洒江南岸，山水悠悠入梦来。"
    },
    {
        "instruction": "将下面这句话翻译成英文",
        "input": "人生如逆旅，我亦是行人。",
        "output": "Life is like a journey against the current; I am just a traveler."
    },
    {
        "instruction": "讲一个冷笑话",
        "input": "",
        "output": "为什么程序员下雨天不带伞？因为他喜欢在云端工作。"
    },
    {
        "instruction": "什么是机器学习？",
        "input": "",
        "output": "机器学习是人工智能的一个分支，它让计算机通过数据自动学习规律，从而完成任务。"
    },
    {
        "instruction": "用Python写一个计算斐波那契数列的函数",
        "input": "",
        "output": "```python\ndef fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)\n```"
    },
    {
        "instruction": "写一段关于月亮的文案",
        "input": "",
        "output": "夜幕降临，月色如水。那轮皎洁的明月，像是一只温柔的眼睛，静静注视着人间的悲欢离合。"
    },
    {
        "instruction": "解释一下牛顿第一定律",
        "input": "",
        "output": "牛顿第一定律指出，物体在不受外力作用时，将保持静止状态或匀速直线运动状态。"
    },
    {
        "instruction": "推荐一本适合入门AI的书籍",
        "input": "",
        "output": "《机器学习实战》是一本适合 AI 入门者的经典书籍，内容通俗易懂，配有丰富代码示例。"
    },
    {
        "instruction": "将下面句子改写得更有文采",
        "input": "我喜欢看日出。",
        "output": "我痴迷于朝阳初升的那一刻，金光洒满天际，仿佛万物皆被唤醒。"
    }
]


# 6. 构造统一的 prompt 文本
def format_prompt(example):
    if example["input"]:
        return f"<s>指令：{example['instruction']}\n输入：{example['input']}\n输出：{example['output']}</s>"
    else:
        return f"<s>指令：{example['instruction']}\n输出：{example['output']}</s>"


for sample in chat_data:
    sample["text"] = format_prompt(sample)

# 7. 转换为 HF Dataset，并分词
dataset = Dataset.from_list(chat_data)


def tokenize(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding


tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# 8. 创建训练配置
training_args = TrainingArguments(
    output_dir="./gpt2-chinese-instruction-lora",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 9. 启动训练
trainer.train()

# 10. 保存模型和分词器
model.save_pretrained("./gpt2-chinese-instruction-lora")
tokenizer.save_pretrained("./gpt2-chinese-instruction-lora")
