import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# === 載入資料與模型 ===
with open("laws_clean.pkl", "rb") as f:
    cleaned_articles = pickle.load(f)

index = faiss.read_index("law_embeddings_ip.faiss")

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16")
selected_model = ""
tokenizer = AutoTokenizer.from_pretrained(selected_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    selected_model, quantization_config=bnb_config, device_map="auto",
    torch_dtype="auto", trust_remote_code=True, low_cpu_mem_usage=True
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.3, do_sample=True)
llm = HuggingFacePipeline(pipeline=pipe)
chat_llm = ChatHuggingFace(llm=llm)

# === 提示詞 ===
ad_prompt = PromptTemplate.from_template("""你是台灣食品藥物管理署的資深法規專家。請嚴格依據法條判斷廣告詞是否違法。

### 相關法條：
{context}

### 廣告詞：
"{ad_text}"

### 判定結果：
【違法判定】：❌ 違法 / ✅ 合法
【違法理由】：
- 違法條文
- 關鍵詞
- 法律風險
【修改建議】：
合法替代用詞
""")

law_prompt = PromptTemplate.from_template("""你是台灣食藥署的法務顧問，請提供法規解釋。

### 相關法條：
{context}

### 問題：
{question}

### 回應格式：
【法條依據】：具體條文
【解釋說明】：詳細說明
【執法標準】：執行依據
【違法後果】：處罰內容
【合規建議】：改善建議
""")

# === 查詢與推論函數 ===
def semantic_search(query, top_k=3):
    vec = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vec)
    scores, indices = index.search(vec, top_k)
    return [cleaned_articles[i] for i in indices[0] if i != -1]

def test_ad_detection(ad_text):
    context = "\n".join(semantic_search(ad_text))
    chain = ad_prompt | chat_llm | StrOutputParser()
    result = chain.invoke({"context": context, "ad_text": ad_text})
    print(f"\n📢 廣告：{ad_text}\n{result}")

def test_law_query(question):
    context = "\n".join(semantic_search(question))
    chain = law_prompt | chat_llm | StrOutputParser()
    result = chain.invoke({"context": context, "question": question})
    print(f"\n❓ 問題：{question}\n{result}")

# === 範例執行 ===
if __name__ == "__main__":
    test_ad_detection("本產品能有效預防癌症！")
    test_law_query("食品廣告不能使用哪些詞彙？")

