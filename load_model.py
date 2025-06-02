"""
本地部署大语言模型
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型路径
model_dir = "/share/llm/Medical_LLM"

# 配置低内存加载
max_memory = {0: "25GiB"}  # 根据你的GPU显存大小调整，这里设置为10GB
offload_folder = "offload"  # 用于存放卸载数据的文件夹

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,  # 启用低CPU内存使用模式
    offload_folder=offload_folder,
    max_memory=max_memory
)
# tokenizer = AutoTokenizer.from_pretrained(model_dir, force_download=False, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 将模型移动到GPU
model = model.to("cuda")  # 假设你使用的是GPU，如果你使用CPU，可以省略这一步

# 保存历史对话
template = "你是一个专业的医院导诊医生，后面你对我多轮症状询问，一次只能问一个问题，你的为AI，我为用户。当你掌握了充足的信息的时候，再做出医学诊断并分析原因，并推荐我去医院哪一个科室，每次回答保持在200字以内，你只能生成ai部分的内容，禁止生成用户的内容，禁止自导自演"
dialog_history = [template]



# 示例输入文本
while(True):
# for i in range(3):
    # arr = ["我有点头晕，你可以根据我的症状进行多轮询问，最后确定我的病因，一次只ww能问一个问题","手脚有点麻木","昨晚喝了凉水"]
    # 获取用户输入
    input_text = input("请输入您的病情（或输入#退出）：")
    input_customer = f"用户：{input_text}"
    dialog_history.append(input_customer)

    # input_text = arr[i]
    # input_text = input("请输入您的病情：")
    # input_text = "昨晚吹空调着凉了，喝了一杯感冒灵不见好，我现在想去医院挂号，但我不知道挂哪个科室，请给我推荐一个最佳科室，并给出理由，并对我的病情进行分析和建议，禁止生成用户内容，禁止自导自演，你只能生成ai表达的内容"
    input_ids = tokenizer.encode(str(dialog_history), return_tensors="pt").to("cuda")  # 将输入也移动到GPU
    # input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
    # 生成文本
    outputs = model.generate(input_ids, max_length=400)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    input_ai = f"AI：{input_text}"
    dialog_history.append(input_ai)

    print(generated_text)
    print()
    print("--"*30)
    print()
    if input_text == "#":
        break




# 初始化对话历史
# dialog_history = ""
# device = "CUDA"
# # while True:
# for i in range(3):
#     arr = ["我有点头晕，你可以根据我的症状进行多轮询问，最后确定我的病因，一次只能问一个问题","手脚有点麻木","昨晚喝了凉水"]
#     # 获取用户输入
#     # input_text = input("请输入您的病情（或输入#退出）：")
#     input_text = arr[i]
#
#
#     # 检查是否退出循环
#     if input_text.strip() == "#":
#         print("对话结束。")
#         break
#
#     # 更新对话历史
#     dialog_history += f"用户: {input_text}\n"
#
#     # 使用tokenizer.__call__方法编码对话历史
#     # inputs = tokenizer(dialog_history, return_tensors="pt", padding=True, truncation=True).to(device)
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")
#     # # 生成回复
#     outputs = model.generate(**inputs, max_length=2000)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#     print(generated_text)
#     print()
#     print("--"*60)
#     print()



    # # 提取并打印AI的回答部分
    # response_start_marker = "AI:"
    # if response_start_marker in generated_text:
    #     ai_response = generated_text.split(response_start_marker)[-1].strip()
    # else:
    #     ai_response = "抱歉，我没有理解您的问题。"
    #
    # print(f"AI: {ai_response}")
    # print("\n" + "--" * 60 + "\n")
    #
    # # 更新对话历史以包含AI的回答
    # dialog_history += f"AI: {ai_response}\n"


# from modelscope import AutoModelForCausalLM, AutoTokenizer

# model_name = "/share/llm/modelsocpe/zpeng1989/COT_Medical_Qwen_Large_Language_Model"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # 输入问题
# question = "请列出所有与葡萄胎相关的症状和体征?"
# inputs = tokenizer(question, return_tensors="pt")

# # 生成答案
# outputs = model.generate(**inputs)
# answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(answer)