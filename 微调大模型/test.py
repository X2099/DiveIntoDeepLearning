# -*- coding: utf-8 -*-
"""
@File    : test.py
@Time    : 2025/6/26 15:25
@Desc    : 
"""

import transformers
import torch
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline

print(transformers.__version__)  # 大多数库的标准方式
print(torch.__version__)  # 大多数库的标准方式

tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
text_generator = TextGenerationPipeline(model, tokenizer)

result = text_generator("这是很久之前的事情了")

print(result)
