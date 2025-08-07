# RAG-LawQA
# 法規查詢輕量 RAG 系統 (Mini RAG for Legal QA)

本專案為「企業級法遵智慧系統」的簡化版本，展示如何使用嵌入模型與輕量大語言模型，實作台灣食品廣告法規的查詢與違規判定。

## 專案特色

- 使用 FAISS 建立法規向量資料庫
- 支援語意搜尋（Semantic Search）
- 使用 HuggingFace 上的輕量模型
- 支援兩大功能：
  - 法規問答查詢
  - 廣告詞違規判斷

## 專案結構

main.py # 主程式：載入向量、進行語意查詢與 LLM 推論
laws_clean.pkl # 清理後的法條 (少量示範用)
law_embeddings_ip.faiss # 對應的 FAISS 索引檔
requirements.txt # 安裝套件清單
README.md # 本說明文件


## 快速開始

### 1. 安裝套件

```bash
pip install -r requirements.txt

2. 執行主程式
python main.py
範例輸入
test_ad_detection("本產品能有效預防癌症！")
test_law_query("食品廣告不能使用哪些詞彙？")





