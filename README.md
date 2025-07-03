import re
import pdfplumber
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
import google.generativeai as genai

# ✅ 1. 設定 Gemini API 金鑰
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # ← 換成你自己的金鑰


# ✅ 2. 定義 Gemini 嵌入函數
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        return embeddings


# ✅ 3. 從 PDF 提取文字
def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text.strip()


# ✅ 4. 將文字分割成語意 chunk
def split_text(text: str, max_chunk_size=500, overlap=100) -> list:
    text = re.sub(r'\s+', ' ', text).strip()
    sentence_delimiters = re.compile(r'(?<=[.!?。！？])\s')
    sentences = sentence_delimiters.split(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            if overlap > 0:
                current_chunk = current_chunk[-overlap:] + sentence + " "
            else:
                current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ✅ 5. 建立與儲存 ChromaDB 資料庫
def create_chroma_db(documents, path="./chroma_db", name="pdf_chunks"):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=path))
    embedding_function = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(name=name, embedding_function=embedding_function)

    ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=ids)

    client.persist()
    return collection


# ✅ 6. 主流程整合
if __name__ == "__main__":
    pdf_path = "your_file.pdf"  # ← 替換成你的 PDF 路徑
    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_text(full_text)

    print(f"📄 讀取完成，共分割為 {len(chunks)} 個文字塊")

    collection = create_chroma_db(chunks, path="./chroma_db", name="example_pdf")

    print(f"✅ 已建立向量資料庫並儲存，共新增 {len(chunks)} 筆嵌入文字")
