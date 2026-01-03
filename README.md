# 基于 RAG 架构的医疗信息智能检索与辅助导诊系统

## 项目简介
本项目基于 [LangChain](https://www.langchain.com/) 框架，构建了一个医疗信息智能检索与辅助导诊系统，采用 **RAG（Retrieval-Augmented Generation）** 架构结合本地大语言模型和向量数据库，实现个性化、精准化的医疗问答和导诊服务。系统可以广泛应用于医院导诊、医生病案查询、辅助诊断等场景，有助于节约医疗成本并提升公共卫生服务质量。

主要功能包括：
- 智能导诊：根据患者症状推荐合适科室。
- 对症约诊：为患者提供就诊指引。
- 精准助诊：帮助医生快速检索病案和历史诊疗信息。
- 多轮问答与记忆：保持上下文一致性，提升问答准确性。

## 项目特色
- **本地化部署**：大语言模型和向量数据库均可本地部署，保障数据安全和隐私。
- **RAG增强**：通过检索增强生成，缓解大模型幻觉问题，提高问答可靠性。
- **多模态文档支持**：支持 txt、pdf、doc、docx 等多种医疗文档格式。
- **动态知识库管理**：可灵活创建、更新和删除不同科室的知识库。
- **前后端分离**：前端提供用户友好的交互界面，后端使用 FastAPI 提供高性能 API 服务。

## 系统架构

系统采用分层架构设计，主要包括：

1. **数据层**
   - 文本处理模块：标准化多源医疗数据，进行语义分割。
   - 向量数据库存储模块：将文本转为高维向量，使用 FAISS 实现高效存储与检索。
   - 动态知识库管理模块：支持知识库全生命周期管理。

2. **检索层**
   - 检索器使用向量相似度 + BM25 加权混合算法，实现高效医疗知识匹配。

3. **模型层**
   - 本地大语言模型（如 huatuoGPT-7B）。
   - RAGChain 完整问答链，包括预处理、向量化、检索、生成和后处理。
   - 多轮问答记忆模块，支持上下文保持。

4. **应用层**
   - 前端：HTML/CSS/JS 构建交互界面。
   - 后端：FastAPI 提供 API 服务，实现医生和患者用户的不同权限管理。

## 技术栈
- Python 3.10+
- [LangChain](https://www.langchain.com/)
- [Transformers](https://huggingface.co/transformers/)
- FAISS 向量数据库
- FastAPI
- 前端：HTML/CSS/JS
- OCR（处理 PDF 医学影像报告）：Tesseract
- 文档解析：python-docx、pdfplumber

## 数据来源
数据集来源于 [Chinese Medical Dialogue Data](https://github.com/Toyhom/Chinese-medical-dialogue-data)：
- 男科（Andriatria）：94,596 问答对
- 内科（IM）：220,606 问答对
- 妇产科（OAGD）：183,751 问答对
- 肿瘤科（Oncology）：75,553 问答对
- 儿科（Pediatric）：101,602 问答对
- 外科（Surgical）：115,991 问答对  
**总计：792,099 条问答数据**

## 安装指南

```bash
# 克隆仓库
git clone <your-repo-url>
cd <your-repo>

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
