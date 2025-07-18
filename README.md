import re
import pdfplumber
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction
import google.generativeai as genai


# ✅ 設定 Gemini API 金鑰
genai.configure(api_key="YOUR_GEMINI_API_KEY")  # ← 替換為你的金鑰


# ✅ Gemini 嵌入函數
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


# ✅ 從 PDF 提取全文
def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text.strip()


# ✅ 將文字分割成語意段落 chunks
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


# ✅ 建立 Chroma 向量資料庫
def create_chroma_db(documents, path="./chroma_db", name="pdf_chunks"):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=path))
    embedding_function = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(name=name, embedding_function=embedding_function)

    ids = [f"doc_{i}" for i in range(len(documents))]
    collection.add(documents=documents, ids=ids)

    client.persist()
    return collection


# ✅ 從 Chroma 查詢最相關段落
def get_relevant_passage(query: str, db, n_results: int = 3) -> list:
    results = db.query(
        query_texts=[query],
        n_results=n_results
    )
    return results["documents"][0]


# ✅ 建立提示語給 Gemini 回答（RAG）
def make_rag_prompt(query: str, relevant_passages: list) -> str:
    context = "\n\n".join(relevant_passages)
    prompt = (
        "Based on the following information:\n\n"
        f"{context}\n\n"
        "Please answer this question:\n"
        f"{query}"
    )
    return prompt


# ✅ 主流程整合
if __name__ == "__main__":
    pdf_path = "your_file.pdf"  # ← 替換你的 PDF 檔案
    full_text = extract_text_from_pdf(pdf_path)
    print("✅ PDF 讀取完成")

    chunks = split_text(full_text)
    print(f"✅ 分割為 {len(chunks)} 個語意段落")

    collection = create_chroma_db(chunks, path="./chroma_db", name="example_pdf")
    print("✅ Chroma 向量資料庫建立完成")

    while True:
        user_question = input("\n❓ 請輸入問題（或輸入 'exit' 結束）：")
        if user_question.lower() == 'exit':
            break

        top_chunks = get_relevant_passage(user_question, collection, n_results=3)
        prompt = make_rag_prompt(user_question, top_chunks)

        print("\n🧠 Gemini Prompt:\n")
        print(prompt)

        response = genai.generate_content(
            model="models/gemini-pro",
            contents=[{"role": "user", "parts": [prompt]}]
        )

        print("\n🤖 Gemini 回答：\n")
        print(response.text.strip())
