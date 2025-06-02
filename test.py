from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import AutoModelForCausalLM
from modelscope.hub.snapshot_download import snapshot_download
import torch
# # 加载微调后的模型

# model = AutoModelForCausalLM.from_pretrained("/share/llm/Medical_LLM",trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained("/share/llm/Medical_LLM",trust_remote_code=True).half().cuda()

model = AutoModelForCausalLM.from_pretrained("/share/llm/Medical_LLM")
tokenizer = AutoTokenizer.from_pretrained("/share/llm/Medical_LLM")


# # 输入问题
question = "请列出所有与葡萄胎相关的症状和体征?"
inputs = tokenizer(question, return_tensors="pt")

# # 生成答案
outputs = model.generate(**inputs)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer)
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # 检查是否有可用的 GPU
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     raise RuntimeError("No GPU available. Please check your environment setup.")

# # 指定模型路径
# model_dir = "/share/llm/modelsocpe/zpeng1989/COT_Medical_Qwen_Large_Language_Model"

# # 加载模型和分词器
# model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
# tokenizer = AutoTokenizer.from_pretrained(model_dir)

# # 将模型移动到 GPU
# model = model.to(device)

# # 输入问题
# question = "请列出所有与葡萄胎相关的症状和体征?"
# inputs = tokenizer(question, return_tensors="pt")

# # 将输入数据移动到 GPU
# inputs = {k: v.to(device) for k, v in inputs.items()}

# # 生成答案
# outputs = model.generate(**inputs)
# answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(answer)
