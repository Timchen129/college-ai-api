
# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# === 1. 初始化與設定 ===
load_dotenv()
# 從環境變數讀取 API KEY
API_KEY = os.getenv("GEMINI_API_KEY") 
genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app) # 允許 WordPress 跨網域存取

# === 2. 模擬資料庫 ===
# 你之後可以在這裡增加更多科系
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
    }
]

# === 3. 向量記憶功能 (RAG) ===
memory = []

def get_embedding(text):
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
    if not memory: return ""
    q_vec = get_embedding(query)
    scored = []
    for text, vec in memory:
        sim = np.dot(q_vec, vec) / (np.linalg.norm(q_vec) * np.linalg.norm(vec))
        scored.append((sim, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    return "\n".join([s[1] for s in scored[:top_k]])

# === 4. 邏輯運算 ===
def calculate_score(student_scores, weights):
    return sum(student_scores.get(s, 0) * weights[s] for s in weights)

def predict_cutoff(data):
    base = data["last_year_score"]
    if data["trend"] == "上升": base += 1.5
    if data["stability"] == "低": base -= 1.0
    return base

# === 5. Gemini AI 分析與格式化 ===
def generate_advice(profile, matches):
    context = retrieve_memory(str(profile))
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    你是一位專業的台灣大學升學顧問。
    學生資料：{profile}
    符合科系：{matches}
    歷史紀錄：{context}
    請給予專業建議，並使用 Markdown 格式（如使用 **粗體** 或 ### 標題）。
    """
    
    response = model.generate_content(prompt)
    advice = response.text
    
    # --- 處理 Markdown 轉 HTML，避免 WordPress 亂碼 ---
    # 1. 處理標題
    advice = re.sub(r'^### (.*$)', r'<h3>\1</h3>', advice, flags=re.M)
    advice = re.sub(r'^## (.*$)', r'<h2>\1</h2>', advice, flags=re.M)
    # 2. 處理粗體
    advice = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', advice)
    # 3. 處理列表
    advice = re.sub(r'^\* (.*$)', r'<li>\1</li>', advice, flags=re.M)
    # 4. 處理換行
    advice = advice.replace("\n", "<br>")
    
    store_memory(f"學生:{profile} | 建議:{advice[:100]}")
    return advice

# === 6. API 路由設定 ===

@app.route("/", methods=["GET"])
def home():
    return "<h1>大腦運作中！</h1><p>API 伺服器已成功啟動。</p>"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        scores = data.get("scores", {})
        profile = data.get("profile", "")

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
            return jsonify({
                "status": "success", 
                "result": "<h3>分析結果</h3>目前的成績可能較難錄取目標科系，建議可以調整志願序或加強弱勢科目。"
            })

        ai_result = generate_advice(profile, matches)
        return jsonify({
            "status": "success", 
            "result": ai_result, 
            "matches": matches
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# === 7. 啟動伺服器 ===
if __name__ == "__main__":
    # 本地測試用，Render 會使用 gunicorn 啟動
    app.run(host="0.0.0.0", port=5000)