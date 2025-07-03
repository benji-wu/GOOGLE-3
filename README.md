import re
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    從 PDF 提取完整文字。
    
    Args:
        pdf_path (str): PDF 檔案路徑。
    
    Returns:
        str: 合併後的所有頁面文字。
    """
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text.strip()


def split_text(text: str, max_chunk_size=500, overlap=100) -> list:
    """
    將輸入文字依語意分割成數個 chunk，以利後續搜尋與檢索。

    Args:
        text (str): 從 PDF 提取的全文字串。
        max_chunk_size (int): 每個 chunk 的最大字元數。
        overlap (int): chunk 間重疊的字元數（保持上下文連貫性）。

    Returns:
        List[str]: 分割後的文字塊列表。
    """
    # 清理多餘空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 以標點符號作為句子分隔依據
    sentence_delimiters = re.compile(r'(?<=[.!?。！？])\s')
    sentences = sentence_delimiters.split(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            # 重疊部分的設置（維持上下文）
            if overlap > 0:
                current_chunk = current_chunk[-overlap:] + sentence + " "
            else:
                current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ✅ 使用範例
if __name__ == "__main__":
    pdf_path = "your_file.pdf"  # ← 請替換為實際 PDF 路徑
    full_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(full_text, max_chunk_size=500, overlap=100)

    for i, chunk in enumerate(text_chunks):
        print(f"--- Chunk {i+1} ---")
        print(chunk)
        print()
