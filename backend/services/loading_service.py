from pypdf import PdfReader
import pdfplumber
import fitz  # PyMuPDF
import logging
import os
from datetime import datetime
import json
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LAParams, LTRect, LTLine
import re

logger = logging.getLogger(__name__)
"""
PDF文档加载服务类
    这个服务类提供了多种PDF文档加载方法，支持不同的加载策略和分块选项。
    主要功能：
"""
class LoadingService:
    """
    PDF文档加载服务类，提供多种PDF文档加载和处理方法。
    
    属性:
        total_pages (int): 当前加载PDF文档的总页数
        current_page_map (list): 存储当前文档的页面映射信息，每个元素包含页面文本和页码
    """
    
    def __init__(self):
        self.total_pages = 0
        self.current_page_map = []
    
    def load_pdf(self, file_path: str, method: str, strategy: str = None, chunking_strategy: str = None, chunking_options: dict = None) -> str:
        """
        加载PDF文档的主方法，支持多种加载策略。

        参数:
            file_path (str): PDF文件路径
            method (str): 加载方法，支持 'pymupdf', 'pypdf', 'pdfplumber', 'unstructured'
            strategy (str, optional): 使用unstructured方法时的策略，可选 'fast', 'hi_res', 'ocr_only'
            chunking_strategy (str, optional): 文本分块策略，可选 'basic', 'by_title'
            chunking_options (dict, optional): 分块选项配置

        返回:
            str: 提取的文本内容
        """
        try:
            if method == "pymupdf":
                return self._load_with_pymupdf(file_path)
            elif method == "pypdf":
                return self._load_with_pypdf(file_path)
            elif method == "pdfplumber":
                return self._load_with_pdfplumber(file_path)
            elif method == "unstructured":
                return self._load_with_unstructured(
                    file_path, 
                    strategy=strategy,
                    chunking_strategy=chunking_strategy,
                    chunking_options=chunking_options
                )
            elif method == "pdfminer":
                return self._load_with_pdfminer(file_path)
            else:
                raise ValueError(f"Unsupported loading method: {method}")
        except Exception as e:
            logger.error(f"Error loading PDF with {method}: {str(e)}")
            raise
    
    def get_total_pages(self) -> int:
        """
        获取当前加载文档的总页数。

        返回:
            int: 文档总页数
        """
        return max(page_data['page'] for page_data in self.current_page_map) if self.current_page_map else 0
    
    def get_page_map(self) -> list:
        """
        获取当前文档的页面映射信息。

        返回:
            list: 包含每页文本内容和页码的列表
        """
        return self.current_page_map
    
    def _load_with_pymupdf(self, file_path: str) -> str:
        """
        使用PyMuPDF库加载PDF文档。
        适合快速处理大量PDF文件，性能最佳。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        text_blocks = []
        try:
            with fitz.open(file_path) as doc:
                self.total_pages = len(doc)
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text("text")
                    if text.strip():
                        text_blocks.append({
                            "text": text.strip(),
                            "page": page_num
                        })
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
        except Exception as e:
            logger.error(f"PyMuPDF error: {str(e)}")
            raise
    
    def _load_with_pypdf(self, file_path: str) -> str:
        """
        使用PyPDF库加载PDF文档。
        适合简单的PDF文本提取，依赖较少。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        try:
            text_blocks = []
            with open(file_path, "rb") as file:
                pdf = PdfReader(file)
                self.total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_blocks.append({
                            "text": page_text.strip(),
                            "page": page_num
                        })
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
        except Exception as e:
            logger.error(f"PyPDF error: {str(e)}")
            raise
    
    def _load_with_unstructured(self, file_path: str, strategy: str = "fast", chunking_strategy: str = "basic", chunking_options: dict = None) -> str:
        """
        使用unstructured库加载PDF文档。
        适合需要更好的文档结构识别和灵活分块策略的场景。

        参数:
            file_path (str): PDF文件路径
            strategy (str): 加载策略，默认'fast'
            chunking_strategy (str): 分块策略，默认'basic'
            chunking_options (dict): 分块选项配置

        返回:
            str: 提取的文本内容
        """
        try:
            # 延迟导入 unstructured 的 partition_pdf，避免在模块导入时触发 NLTK 的下载/检查，
            # 导入失败时记录错误并抛出，以便调用方能收到明确的异常而不是整个应用崩溃。
            try:
                from unstructured.partition.pdf import partition_pdf
            except Exception as e:
                logger.error(f"Failed to import unstructured.partition.pdf: {e}")
                raise

            strategy_params = {
                "fast": {"strategy": "fast"},
                "hi_res": {"strategy": "hi_res"},
                "ocr_only": {"strategy": "ocr_only"}
            }            
         
            # Prepare chunking parameters based on strategy
            chunking_params = {}
            if chunking_strategy == "basic":
                chunking_params = {
                    "max_characters": chunking_options.get("maxCharacters", 4000),
                    "new_after_n_chars": chunking_options.get("newAfterNChars", 3000),
                    "combine_text_under_n_chars": chunking_options.get("combineTextUnderNChars", 2000),
                    "overlap": chunking_options.get("overlap", 200),
                    "overlap_all": chunking_options.get("overlapAll", False)
                }
            elif chunking_strategy == "by_title":
                chunking_params = {
                    "chunking_strategy": "by_title",
                    "combine_text_under_n_chars": chunking_options.get("combineTextUnderNChars", 2000),
                    "multipage_sections": chunking_options.get("multiPageSections", False)
                }

            # Combine strategy parameters with chunking parameters
            params = {**strategy_params.get(strategy, {"strategy": "fast"}), **chunking_params}

            elements = partition_pdf(file_path, **params)

            # Add debug logging
            for elem in elements:
                logger.debug(f"Element type: {type(elem)}")
                logger.debug(f"Element content: {str(elem)}")
                logger.debug(f"Element dir: {dir(elem)}")

            text_blocks = []
            pages = set()

            for elem in elements:
                metadata = elem.metadata.__dict__
                page_number = metadata.get('page_number')

                if page_number is not None:
                    pages.add(page_number)

                    # Convert element to a serializable format
                    cleaned_metadata = {}
                    for key, value in metadata.items():
                        if key == '_known_field_names':
                            continue

                        try:
                            # Try JSON serialization to test if value is serializable
                            json.dumps({key: value})
                            cleaned_metadata[key] = value
                        except (TypeError, OverflowError):
                            # If not serializable, convert to string
                            cleaned_metadata[key] = str(value)

                    # Add additional element information
                    cleaned_metadata['element_type'] = elem.__class__.__name__
                    cleaned_metadata['id'] = str(getattr(elem, 'id', None))
                    cleaned_metadata['category'] = str(getattr(elem, 'category', None))

                    text_blocks.append({
                        "text": str(elem),
                        "page": page_number,
                        "metadata": cleaned_metadata
                    })

            self.total_pages = max(pages) if pages else 0
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
            
        except Exception as e:
            logger.error(f"Unstructured error: {str(e)}")
            raise

    def _load_with_pdfplumber(self, file_path: str) -> str:
        """
        使用pdfplumber库加载PDF文档。
        适合需要处理表格或需要文本位置信息的场景。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        text_blocks = []
        try:
            with pdfplumber.open(file_path) as pdf:
                self.total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_blocks.append({
                            "text": page_text.strip(),
                            "page": page_num
                        })
            self.current_page_map = text_blocks
            return "\n".join(block["text"] for block in text_blocks)
        except Exception as e:
            logger.error(f"pdfplumber error: {str(e)}")
            raise
    
    def save_document(self, filename: str, chunks: list, metadata: dict, loading_method: str, strategy: str = None, chunking_strategy: str = None) -> str:
        """
        保存处理后的文档数据。

        参数:
            filename (str): 原PDF文件名
            chunks (list): 文档分块列表
            metadata (dict): 文档元数据
            loading_method (str): 使用的加载方法
            strategy (str, optional): 使用的加载策略
            chunking_strategy (str, optional): 使用的分块策略

        返回:
            str: 保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            base_name = filename.replace('.pdf', '').split('_')[0]
            
            # Adjust the document name to include strategy if unstructured
            if loading_method == "unstructured" and strategy:
                doc_name = f"{base_name}_{loading_method}_{strategy}_{chunking_strategy}_{timestamp}"
            else:
                doc_name = f"{base_name}_{loading_method}_{timestamp}"
            
            # 构建文档数据结构，确保所有值都是可序列化的
            document_data = {
                "filename": str(filename),
                "total_chunks": int(len(chunks)),
                "total_pages": int(metadata.get("total_pages", 1)),
                "loading_method": str(loading_method),
                "loading_strategy": str(strategy) if loading_method == "unstructured" and strategy else None,
                "chunking_strategy": str(chunking_strategy) if loading_method == "unstructured" and chunking_strategy else None,
                "chunking_method": "loaded",
                "timestamp": datetime.now().isoformat(),
                "chunks": chunks
            }
            
            # 保存到文件
            filepath = os.path.join("01-loaded-docs", f"{doc_name}.json")
            os.makedirs("01-loaded-docs", exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, ensure_ascii=False, indent=2)
                
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")
            raise

    def extract_pdf_tables(self, pdf_path):
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
        
        return result_strings

    def _load_with_pdfminer(self, file_path: str) -> str:
        
        """
        使用pdfminer库加载PDF文档。

        参数:
            file_path (str): PDF文件路径

        返回:
            str: 提取的文本内容
        """
        text_by_page = {}
        text_blocks = []
        # 遍历所有页面
        for page_num, page_layout in enumerate(extract_pages(file_path)):
            page_text = ""
            # 遍历页面中的每个文本容器
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    # 提取文本
                    page_text += element.get_text()
            text_blocks.append({
                "text": page_text.strip(),
                "page": page_num + 1,
                "metadata": "text"
            })

        tables = self.extract_pdf_tables(file_path)

        table_row = 0

        for table in tables:
            text_blocks.append({
                "text": table.strip(),
                "page": table_row + 1,
                "metadata": "table"
            })
            table_row += 1

        self.total_pages = page_num + 1
        self.current_page_map.extend(text_blocks)
        text_by_page = "\n".join(block["text"] for block in text_blocks)
        return text_by_page
