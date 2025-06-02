"""
##############################实现功能####################################
# 本文件用于PDF文件转化为TXT文件
###############################函数######################################
# extract_text_from_pdf(filename, page_numbers=None, min_line_length=1)
#   参数：filename：PDF文件名或者地址
#        page_numbers：需要提取的页码，默认为None，表示提取所有页
#        min_line_length：最小行长度，默认为1
#   返回：提取的String文本
--------------------------------------------------------------------------
"""

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

##提取文本函数S
def extract_text_from_pdf(filename, page_numbers=None, min_line_length=1):
    '''从 PDF 文件中（按指定页码）提取文字'''
    paragraphs = []
    buffer = ''
    full_text = ''
    # 提取全部文本
    for i, page_layout in enumerate(extract_pages(filename)):
        # 如果指定了页码范围，跳过范围外的页
        if page_numbers is not None and i not in page_numbers:
            continue
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                full_text += element.get_text() + '\n'
    # 按空行分隔，将文本重新组织成段落
    lines = full_text.split('\n')
    for text in lines:
        if len(text) >= min_line_length:
            buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
        elif buffer:
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs

paragraphs = extract_text_from_pdf("/share/llm/database/original_data/临床诊断学.pdf", min_line_length=10)
# 定义保存文件的路径
output_file_path = "/data/xms/Rag_LLM/database_medical/临床诊断学.txt"

# 将文本保存到文件
with open(output_file_path, "w", encoding="utf-8") as file:
    for paragraph in paragraphs:
        file.write(paragraph + "\n")

print(f"文本已成功保存到 {output_file_path}")

for para in paragraphs[:4]:
    print(para + "\n")
if __name__ == "main":
    #测试输出文本效果
    paragraphs = extract_text_from_pdf("/share/llm/database/original_data/临床诊断学", min_line_length=10)
    for para in paragraphs[:4]:
        print(para + "\n")