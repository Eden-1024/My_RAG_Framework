
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTLine
import re
import os
import sys

def extract_pdf_tables(pdf_path):
    """
    从PDF中提取表格数据，返回制表符分隔的字符串
    格式：\t 内容1 \t 内容2 \t ...
    """
    
    # 存储所有页面的表格数据
    all_tables_data = []
    
    # 提取页面布局信息
    for page_layout in extract_pages(pdf_path):
        page_tables = []
        
        # 收集页面中的所有文本元素和线框元素
        text_elements = []
        line_elements = []
        
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                # 获取文本的精确位置和内容
                text = element.get_text().strip()
                if text:  # 只处理非空文本
                    x0, y0, x1, y1 = element.bbox
                    text_elements.append({
                        'text': text,
                        'x0': x0,
                        'y0': y0,
                        'x1': x1,
                        'y1': y1,
                        'page_height': page_layout.height
                    })
            elif isinstance(element, LTRect) or isinstance(element, LTLine):
                # 收集表格线框
                x0, y0, x1, y1 = element.bbox
                line_elements.append({
                    'type': 'rect' if isinstance(element, LTRect) else 'line',
                    'x0': x0,
                    'y0': y0,
                    'x1': x1,
                    'y1': y1
                })
        
        if not text_elements:
            continue
        
        # 根据Y坐标对文本进行分组（行）
        # 先将文本按Y坐标排序（从上到下）
        text_elements.sort(key=lambda x: -x['y0'])  # 负号表示从上到下
        
        # 识别行：Y坐标相近的文本归为同一行
        rows = []
        current_row = []
        row_y = None
        row_threshold = 5  # Y坐标相差小于5像素的视为同一行
        
        for text in text_elements:
            if row_y is None:
                row_y = text['y0']
                current_row.append(text)
            elif abs(text['y0'] - row_y) < row_threshold:
                current_row.append(text)
            else:
                # 当前行结束，开始新行
                if current_row:
                    rows.append(current_row)
                current_row = [text]
                row_y = text['y0']
        
        if current_row:
            rows.append(current_row)
        
        # 对每行内的文本按X坐标排序（从左到右）
        for row in rows:
            row.sort(key=lambda x: x['x0'])
        
        # 转换为表格格式
        for row in rows:
            row_data = []
            for cell in row:
                # 清理文本：移除多余的空格和换行
                clean_text = re.sub(r'\s+', ' ', cell['text']).strip()
                if clean_text:
                    row_data.append(clean_text)
            
            if row_data:  # 只处理非空行
                page_tables.append(row_data)
        
        if page_tables:
            all_tables_data.extend(page_tables)
    
    # 将表格数据转换为所需格式的字符串
    result_strings = []
    for row in all_tables_data:
        # 使用制表符分隔每个单元格，并在前后添加\t
        row_str = "\t " + " \t ".join(row) + " \t"
        result_strings.append(row_str)
    
    return "\n".join(result_strings)


def extract_pdf_tables_enhanced(pdf_path, min_column_width=20):
    """
    增强版的表格提取函数，尝试识别列结构
    """
    
    all_tables_data = []
    
    for page_layout in extract_pages(pdf_path):
        # 收集所有文本元素
        text_elements = []
        
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text = element.get_text().strip()
                if text:
                    x0, y0, x1, y1 = element.bbox
                    text_elements.append({
                        'text': text,
                        'x0': x0,
                        'y0': y0,
                        'x1': x1,
                        'y1': y1
                    })
        
        if not text_elements:
            continue
        
        # 按Y坐标分组（行）
        text_elements.sort(key=lambda x: -x['y0'])  # 从上到下
        
        # 更智能的行分组
        rows = []
        current_row = []
        
        for i, text in enumerate(text_elements):
            if not current_row:
                current_row.append(text)
            else:
                # 检查是否与上一文本在同一行
                last_text = current_row[-1]
                y_diff = abs(text['y0'] - last_text['y0'])
                
                # 如果Y坐标相近，且不在同一垂直位置，可能是同一行的不同列
                if y_diff < 8:  # 8像素的阈值
                    current_row.append(text)
                else:
                    # 新行开始
                    rows.append(current_row)
                    current_row = [text]
        
        if current_row:
            rows.append(current_row)
        
        # 识别列边界
        column_boundaries = []
        for row in rows:
            for text in row:
                if not column_boundaries:
                    column_boundaries.append([text['x0'], text['x1']])
                else:
                    found = False
                    for col in column_boundaries:
                        # 如果x0在现有列范围内，扩展列边界
                        if text['x0'] >= col[0] - 5 and text['x0'] <= col[1] + 5:
                            col[0] = min(col[0], text['x0'])
                            col[1] = max(col[1], text['x1'])
                            found = True
                            break
                    if not found:
                        column_boundaries.append([text['x0'], text['x1']])
        
        # 合并重叠的列
        column_boundaries.sort(key=lambda x: x[0])
        merged_columns = []
        for col in column_boundaries:
            if not merged_columns:
                merged_columns.append(col)
            else:
                last_col = merged_columns[-1]
                if col[0] <= last_col[1] + 10:  # 有重叠
                    last_col[1] = max(last_col[1], col[1])
                else:
                    merged_columns.append(col)
        
        # 根据列边界整理数据
        processed_rows = []
        for row in rows:
            row.sort(key=lambda x: x['x0'])
            row_data = [''] * len(merged_columns)
            
            for text in row:
                # 找到文本属于哪一列
                for i, col in enumerate(merged_columns):
                    if text['x0'] >= col[0] - 5 and text['x1'] <= col[1] + 5:
                        if row_data[i]:
                            row_data[i] += ' ' + text['text']
                        else:
                            row_data[i] = text['text']
                        break
            
            # 清理空列
            row_data = [cell.strip() for cell in row_data if cell.strip()]
            if row_data:
                processed_rows.append(row_data)
        
        all_tables_data.extend(processed_rows)
    
    # 转换为目标格式
    result_strings = []
    for row in all_tables_data:
        row_str = "\t " + " \t ".join(row) + " \t"
        result_strings.append(row_str)
    
    return "\n".join(result_strings)


# 使用示例
if __name__ == "__main__":
    file_path = "/Users/eden/Documents/git/rag-in-action-master/90-文档-Data/复杂PDF/billionaires_page-1-5.pdf"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        print(f"当前工作目录: {os.getcwd()}")
        print("请确保:")
        print("1. 在项目根目录运行脚本")
        print("2. PDF文件路径正确")
        sys.exit(1)

    print(f"正在处理文件: {file_path}")
    # 基本使用
    pdf_file = file_path  # 替换为你的PDF文件路径

    try:
        print("=== 基本表格提取 ===")
        result_basic = extract_pdf_tables(pdf_file)
        print(result_basic)
        
        print("\n=== 增强版表格提取 ===")
        result_enhanced = extract_pdf_tables_enhanced(pdf_file)
        print(result_enhanced)
        
        # 保存到文件
        with open("extracted_tables.txt", "w", encoding="utf-8") as f:
            f.write("=== 提取的表格数据 ===\n")
            f.write(result_enhanced)
            
    except FileNotFoundError:
        print(f"错误：找不到文件 {pdf_file}")
    except Exception as e:
        print(f"提取过程中发生错误: {str(e)}")