#SDK模型下载
from modelscope import snapshot_download

# 指定下载路径
model_dir = snapshot_download(
    'zpeng1989/COT_Medical_Qwen_Large_Language_Model',
    cache_dir='/share/llm/modelsocpe'
)

print(f"模型已下载到: {model_dir}")