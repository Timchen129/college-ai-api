# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 23:30:43 2026

@author: yiting
"""
import os
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# === 1. 初始化與設定 ===
load_dotenv()
# 請確保 .env 檔案中有 GEMINI_API_KEY
genai.configure(api_key=os.getenv("AIzaSyB560GCDjFKSJ5B82wKZ4HvJH81osvcgF8"))

app = Flask(__name__)

# === 2. 模擬資料庫 (建議之後改用 SQLite) ===
majors_db = [
    {
        "school": "台灣大學",
        "major": "電機工程",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "last_year_score": 58,
        "trend": "上升",
        "stability": "高",
        "required_abilities": ["數學", "物理"],
        "salary": "高",
        "career": ["半導體", "IC設計"]
    },
    {
        "school": "清華大學",
        "major": "資訊工程",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "last_year_score": 56,
        "trend": "上升",
        "stability": "高",
        "required_abilities": ["數學", "邏輯"],
        "salary": "高",
        "career": ["軟體", "AI"]
    }
]

# === 3. 向量記憶功能 (RAG) ===
memory = []

def get_embedding(text):
    """使用 Gemini 的模型產生向量"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(result['embedding'])

def store_memory(text):
    vec = get_embedding(text)
    memory.append((text, vec))

def retrieve_memory(query, top_k=2):
    if not memory:
        return ""
    q_vec = get_embedding(query)
    scored = []
    for text, vec in memory:
        # 計算餘弦相似度 (Cosine Similarity)
        sim = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
        scored.append((sim, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n".join([s[1] for s in scored[:top_k]])

# === 4. 邏輯運算 (分數與預測) ===
def calculate_score(student_scores, weights):
    return sum(student_scores.get(s, 0) * weights[s] for s in weights)

def predict_cutoff(data):
    base = data["last_year_score"]
    # 簡單的邏輯預測
    if data["trend"] == "上升": base += 1.5
    if data["stability"] == "低": base -= 1.0
    return base

# === 5. Gemini AI 分析 ===
def generate_advice(profile, matches):
    # 取得歷史記憶背景
    context = retrieve_memory(str(profile))
    
    # 初始化 Gemini Pro 模型
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
你是一位專業的台灣大學升學顧問，請針對以下資料進行務實分析：

【學生個人特質與需求】：
{profile}

【系統初步篩選出的合格科系】：
{matches}

【過往諮詢參考紀錄】：
{context}

請提供以下分析回饋：
1. 推薦最適合的 2-3 個科系並說明原因。
2. 哪些科系雖然分數到了，但可能與學生能力不符？（風險評估）
3. 針對未來的就業趨勢給予具體建議。
4. 回話風格要冷靜、客觀，不要給予空洞的鼓勵。
"""
    
    response = model.generate_content(prompt)
    advice = response.text
    
    # 存入記憶，方便下次參考
    store_memory(f"學生:{profile} | 建議:{advice[:100]}")
    
    return advice

# === 6. API 路由設定 ===
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        scores = data.get("scores")
        profile = data.get("profile")

        # 找出符合門檻的科系
        matches = []
        for m in majors_db:
            student_total = calculate_score(scores, m["weights"])
            target_cutoff = predict_cutoff(m)
            if student_total >= target_cutoff:
                matches.append({
                    "school": m["school"],
                    "major": m["major"],
                    "score": student_total
                })

        if not matches:
            return jsonify({"result": "目前的成績可能較難錄取目標科系，建議擴大志願範圍。"})

        # 呼叫 Gemini 進行分析
        ai_result = generate_advice(profile, matches)

        return jsonify({
            "status": "success",
            "result": ai_result,
            "matches": matches
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # 本地測試時啟動
    app.run(host="0.0.0.0", port=5000, debug=True)
