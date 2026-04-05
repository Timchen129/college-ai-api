
import os
import re
import numpy as np
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# === 初始化 ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)

# === 科系資料（升級版）===
majors_db = [
    {
        "school": "台灣大學",
        "major": "電機工程",
        "weights": {"國文":1,"英文":2,"數學":3,"自然":3,"社會":0},
        "last_year_score": 58,
        "trend": "上升",
        "stability": "高",
        "required_abilities": ["數學","物理"],
        "bonus_factor": ["英文"],
        "salary": "高",
        "career": ["半導體","IC設計"]
    },
    {
        "school": "清華大學",
        "major": "資訊工程",
        "weights": {"國文":1,"英文":2,"數學":3,"自然":2,"社會":0},
        "last_year_score": 56,
        "trend": "上升",
        "stability": "高",
        "required_abilities": ["數學","邏輯"],
        "bonus_factor": ["英文"],
        "salary": "高",
        "career": ["AI","軟體"]
    },
    {
        "school": "私立大學",
        "major": "商業管理",
        "weights": {"國文":2,"英文":2,"數學":1,"自然":0,"社會":2},
        "last_year_score": 45,
        "trend": "下降",
        "stability": "低",
        "required_abilities": ["溝通"],
        "bonus_factor": [],
        "salary": "中",
        "career": ["行銷","業務"]
    }
]

# === 向量記憶 ===
memory = []

def get_embedding(text):
    res = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(res['embedding'])

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def store_memory(text):
    memory.append((text, get_embedding(text)))

def retrieve_memory(query, top_k=3):
    if not memory:
        return ""
    q_vec = get_embedding(query)
    scored = [(cosine_sim(q_vec, vec), text) for text, vec in memory]
    scored.sort(reverse=True)
    return "\n".join([t for _, t in scored[:top_k]])

# === 分數邏輯 ===
def calculate_score(scores, weights):
    return sum(scores.get(s, 0) * weights[s] for s in weights)

def predict_cutoff(m):
    base = m["last_year_score"]

    if m["trend"] == "上升":
        base += 2
    elif m["trend"] == "下降":
        base -= 2

    if m["stability"] == "高":
        base += 1
    elif m["stability"] == "低":
        base -= 1

    return base

def match_majors(scores):
    results = []

    for m in majors_db:
        student_score = calculate_score(scores, m["weights"])
        cutoff = predict_cutoff(m)

        if student_score >= cutoff:
            results.append({
                "school": m["school"],
                "major": m["major"],
                "score": student_score,
                "cutoff": cutoff,
                "data": m
            })

    return results

# === AI 分析 ===
def generate_advice(profile, matches):

    context = retrieve_memory(str(profile))

    abroad = profile.get("出國", "n")
    abroad_text = "請提供國外升學與發展建議" if abroad == "y" else "不用提供國外建議"

    prompt = f"""
你是一位非常務實的台灣升學顧問。

【學生資料】
{profile}

【符合科系】
{matches}

【歷史參考】
{context}

請分析：

1️⃣ 推薦科系（最多3個）
2️⃣ 能力是否符合（比興趣重要）
3️⃣ 哪些只是興趣符合但風險高
4️⃣ 分數是否穩上或邊緣
5️⃣ 英文是否為關鍵決勝點
6️⃣ 未來出路（薪資＋產業）
7️⃣ {abroad_text}

規則：
- 不要理想化
- 不要空話
- 要像真實升學顧問
- 用條列式
"""

    model = genai.GenerativeModel('gemini-1.5-flash')
    res = model.generate_content(prompt)
    text = res.text

    # === Markdown → HTML ===
    text = re.sub(r'^### (.*$)', r'<h3>\1</h3>', text, flags=re.M)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'^\* (.*$)', r'<li>\1</li>', text, flags=re.M)
    text = text.replace("\n", "<br>")

    store_memory(f"{profile} => {text[:100]}")

    return text

# === API ===
@app.route("/")
def home():
    return "<h1>AI升學系統運作中</h1>"

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json

        scores = data.get("scores", {})
        profile = data.get("profile", {})

        matches = match_majors(scores)

        if not matches:
            return jsonify({
                "status": "success",
                "result": "<h3>分析結果</h3>目前分數較難錄取，建議調整策略"
            })

        result = generate_advice(profile, matches)

        return jsonify({
            "status": "success",
            "result": result,
            "matches": matches
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# === 啟動 ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)