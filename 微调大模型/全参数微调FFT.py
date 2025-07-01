import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

from train_data import chat_data

# ----------------------------
# 1. 自动检测设备 (GPU 优先，否则 CPU)
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前设备：{device}")

# ----------------------------
# 2. 加载模型和分词器
# ----------------------------
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# ----------------------------
# 3. 添加特殊token，扩充词表
# ----------------------------
special_tokens_dict = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "additional_special_tokens": ["<user>", "</user>", "<bot>", "</bot>"]
}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))


# ------------------------------------------------------------------
# 调用你微调后的中文 GPT 模型，根据用户输入的提示（prompt）生成对话回复
# ------------------------------------------------------------------
def chat(prompt, max_new_tokens=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.convert_tokens_to_ids("</bot>"),
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False,  # 贪心解码，最保守
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


prompt = "<user>你是谁？</user><bot>"
chat(prompt)

# ----------------------------
# 4. 使用 Dataset 加载数据
# ----------------------------
dataset = Dataset.from_list(chat_data)


# ----------------------------
# 5. 分词和标签处理，确保padding和truncation
# ----------------------------
def tokenize_function(example):
    encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding


tokenized_dataset = dataset.map(tokenize_function)

# ----------------------------
# 6. 数据整理器，自动对齐输入和标签
# ----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ----------------------------
# 7. 训练参数设置
# ----------------------------
training_args = TrainingArguments(
    output_dir="./chatgpt2-chinese",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=50,
    fp16=True if device == "cuda" else False,  # 仅GPU时启用混合精度
    report_to="none"
)

# ----------------------------
# 8. Trainer 初始化
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# ----------------------------
# 9. 启动训练
# ----------------------------
trainer.train()

# ----------------------------
# 10. 保存模型和分词器
# ----------------------------
trainer.save_model("./chatgpt2-chinese")
tokenizer.save_pretrained("./chatgpt2-chinese")
