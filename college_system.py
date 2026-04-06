import os
import re
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# === 初始化 ===
load_dotenv()

MODEL_NAME = "gemini-2.5-flash"   # ← 唯一需要改模型的地方

try:
    import google.generativeai as genai
    print(f"Current Key Check: {os.getenv('GEMINI_API_KEY')[:10]}...")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# ============================================================
# 快取層（減少重複 API 呼叫，節省費用）
# ============================================================

_cache = {}
CACHE_TTL = 3600  # 1 小時

def cache_get(key):
    item = _cache.get(key)
    if item and time.time() - item["ts"] < CACHE_TTL:
        return item["val"]
    return None

def cache_set(key, val):
    _cache[key] = {"val": val, "ts": time.time()}

def make_cache_key(*args):
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

# ============================================================
# 科系資料庫（擴充版）
# ============================================================

majors_db = [
    # ── 頂大理工 ──
    {
        "school": "台灣大學", "major": "電機工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "last_year_score": 58, "trend": "上升", "stability": "高",
        "required_abilities": ["數學", "物理"], "bonus_factor": ["英文"],
        "salary_avg": 85000, "career": ["半導體", "IC設計", "AI工程"],
        "employment_rate": 97, "overseas_ratio": 35,
        "industry_outlook": "正面",
        "ai_impact": "受益",
        "notes": "台積電、聯發科主要來源校系",
        "license_required": False,
        "double_major_friendly": True,
    },
    {
        "school": "台灣大學", "major": "資訊工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "last_year_score": 60, "trend": "上升", "stability": "高",
        "required_abilities": ["數學", "邏輯"], "bonus_factor": ["英文"],
        "salary_avg": 90000, "career": ["AI", "軟體工程", "新創"],
        "employment_rate": 98, "overseas_ratio": 40,
        "industry_outlook": "正面",
        "ai_impact": "受益",
        "notes": "矽谷就業率最高的台灣科系之一",
        "license_required": False,
        "double_major_friendly": True,
    },
    {
        "school": "清華大學", "major": "資訊工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "last_year_score": 56, "trend": "上升", "stability": "高",
        "required_abilities": ["數學", "邏輯"], "bonus_factor": ["英文"],
        "salary_avg": 82000, "career": ["AI", "軟體", "半導體"],
        "employment_rate": 96, "overseas_ratio": 30,
        "industry_outlook": "正面",
        "ai_impact": "受益",
        "notes": "竹科地緣優勢明顯",
        "license_required": False,
        "double_major_friendly": True,
    },
    {
        "school": "清華大學", "major": "電機工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "last_year_score": 55, "trend": "持平", "stability": "高",
        "required_abilities": ["數學", "物理"], "bonus_factor": [],
        "salary_avg": 80000, "career": ["半導體", "IC設計", "電力電子"],
        "employment_rate": 96, "overseas_ratio": 28,
        "industry_outlook": "正面",
        "ai_impact": "受益",
        "notes": "",
        "license_required": False,
        "double_major_friendly": True,
    },
    {
        "school": "交通大學", "major": "資訊工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "last_year_score": 54, "trend": "上升", "stability": "高",
        "required_abilities": ["數學", "邏輯"], "bonus_factor": ["英文"],
        "salary_avg": 80000, "career": ["AI", "網路", "軟體"],
        "employment_rate": 95, "overseas_ratio": 28,
        "industry_outlook": "正面",
        "ai_impact": "受益",
        "notes": "與清華合校後資源更豐富",
        "license_required": False,
        "double_major_friendly": True,
    },
    {
        "school": "成功大學", "major": "電機工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "last_year_score": 52, "trend": "持平", "stability": "高",
        "required_abilities": ["數學", "物理"], "bonus_factor": [],
        "salary_avg": 75000, "career": ["半導體", "電子業", "IC設計"],
        "employment_rate": 95, "overseas_ratio": 22,
        "industry_outlook": "正面",
        "ai_impact": "受益",
        "notes": "南科人才搖籃，就業在地化優勢",
        "license_required": False,
        "double_major_friendly": True,
    },
    # ── 醫學 ──
    {
        "school": "成功大學", "major": "醫學系", "group": "醫",
        "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
        "last_year_score": 65, "trend": "持平", "stability": "高",
        "required_abilities": ["記憶力", "自然"], "bonus_factor": [],
        "salary_avg": 120000, "career": ["醫師", "研究醫學"],
        "employment_rate": 99, "overseas_ratio": 5,
        "industry_outlook": "穩定",
        "ai_impact": "部分影響",
        "notes": "南台灣最頂尖醫學系，錄取分數極高",
        "license_required": True,
        "double_major_friendly": False,
    },
    {
        "school": "陽明交大", "major": "醫學系", "group": "醫",
        "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
        "last_year_score": 67, "trend": "持平", "stability": "高",
        "required_abilities": ["記憶力", "自然"], "bonus_factor": [],
        "salary_avg": 125000, "career": ["醫師", "研究醫學"],
        "employment_rate": 99, "overseas_ratio": 8,
        "industry_outlook": "穩定",
        "ai_impact": "部分影響",
        "notes": "與台大醫學系並列最頂尖",
        "license_required": True,
        "double_major_friendly": False,
    },
    {
        "school": "台灣大學", "major": "護理系", "group": "醫",
        "weights": {"國文": 1, "英文": 1, "數學": 1, "自然": 2, "社會": 1},
        "last_year_score": 48, "trend": "上升", "stability": "高",
        "required_abilities": ["自然", "耐心"], "bonus_factor": [],
        "salary_avg": 65000, "career": ["護理師", "護理長", "醫療管理"],
        "employment_rate": 98, "overseas_ratio": 10,
        "industry_outlook": "正面",
        "ai_impact": "低影響",
        "notes": "護理缺工嚴重，薪資近年大幅提升",
        "license_required": True,
        "double_major_friendly": False,
    },
    {
        "school": "台灣大學", "major": "藥學系", "group": "醫",
        "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
        "last_year_score": 58, "trend": "持平", "stability": "高",
        "required_abilities": ["化學", "自然"], "bonus_factor": [],
        "salary_avg": 75000, "career": ["藥師", "製藥研發", "臨床試驗"],
        "employment_rate": 96, "overseas_ratio": 12,
        "industry_outlook": "穩定",
        "ai_impact": "部分影響",
        "notes": "藥師執照保障穩定就業",
        "license_required": True,
        "double_major_friendly": False,
    },
    # ── 社會法商 ──
    {
        "school": "台灣大學", "major": "法律系", "group": "社會",
        "weights": {"國文": 2, "英文": 2, "數學": 1, "自然": 0, "社會": 3},
        "last_year_score": 57, "trend": "持平", "stability": "中",
        "required_abilities": ["閱讀", "邏輯"], "bonus_factor": ["國文"],
        "salary_avg": 70000, "career": ["律師", "法務", "公職"],
        "employment_rate": 88, "overseas_ratio": 15,
        "industry_outlook": "穩定",
        "ai_impact": "部分影響",
        "notes": "需考律師證照才能發揮最大效益",
        "license_required": True,
        "double_major_friendly": True,
    },
    {
        "school": "台灣大學", "major": "經濟系", "group": "社會",
        "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 0, "社會": 2},
        "last_year_score": 55, "trend": "上升", "stability": "中",
        "required_abilities": ["數學", "分析"], "bonus_factor": ["英文"],
        "salary_avg": 72000, "career": ["金融", "顧問", "政策研究"],
        "employment_rate": 90, "overseas_ratio": 25,
        "industry_outlook": "穩定",
        "ai_impact": "部分影響",
        "notes": "出國深造比例高，適合有研究所規劃者",
        "license_required": False,
        "double_major_friendly": True,
    },
    {
        "school": "政治大學", "major": "財務管理", "group": "社會",
        "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 0, "社會": 2},
        "last_year_score": 50, "trend": "持平", "stability": "中",
        "required_abilities": ["數學", "分析"], "bonus_factor": ["英文"],
        "salary_avg": 65000, "career": ["投資銀行", "基金", "企業財務"],
        "employment_rate": 89, "overseas_ratio": 20,
        "industry_outlook": "穩定",
        "ai_impact": "部分影響",
        "notes": "政大商院品牌效益強，金融業認可度高",
        "license_required": False,
        "double_major_friendly": True,
    },
    # ── 人文 ──
    {
        "school": "台灣大學", "major": "外文系", "group": "人文",
        "weights": {"國文": 2, "英文": 3, "數學": 0, "自然": 0, "社會": 1},
        "last_year_score": 50, "trend": "下降", "stability": "低",
        "required_abilities": ["英文", "語言"], "bonus_factor": [],
        "salary_avg": 42000, "career": ["翻譯", "外貿", "教師"],
        "employment_rate": 75, "overseas_ratio": 20,
        "industry_outlook": "負面",
        "ai_impact": "高度衝擊",
        "notes": "AI翻譯工具衝擊傳統翻譯市場，需發展複合能力",
        "license_required": False,
        "double_major_friendly": True,
    },
    # ── 理學 ──
    {
        "school": "台灣大學", "major": "數學系", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "last_year_score": 52, "trend": "上升", "stability": "中",
        "required_abilities": ["數學", "抽象思維"], "bonus_factor": [],
        "salary_avg": 70000, "career": ["金融工程", "AI研究", "精算"],
        "employment_rate": 88, "overseas_ratio": 35,
        "industry_outlook": "正面",
        "ai_impact": "受益",
        "notes": "AI時代數學底子是核心競爭力，轉資工/金融門票",
        "license_required": False,
        "double_major_friendly": True,
    },
    # ── 私立 ──
    {
        "school": "私立大學", "major": "資訊工程", "group": "理工",
        "weights": {"國文": 1, "英文": 1, "數學": 2, "自然": 1, "社會": 0},
        "last_year_score": 40, "trend": "下降", "stability": "低",
        "required_abilities": ["數學"], "bonus_factor": [],
        "salary_avg": 55000, "career": ["軟體開發", "IT維運"],
        "employment_rate": 82, "overseas_ratio": 5,
        "industry_outlook": "負面",
        "ai_impact": "高度衝擊",
        "notes": "AI工具普及後，非頂尖資工系基層就業壓力大",
        "license_required": False,
        "double_major_friendly": False,
    },
    {
        "school": "私立大學", "major": "商業管理", "group": "社會",
        "weights": {"國文": 2, "英文": 2, "數學": 1, "自然": 0, "社會": 2},
        "last_year_score": 42, "trend": "下降", "stability": "低",
        "required_abilities": ["溝通"], "bonus_factor": [],
        "salary_avg": 45000, "career": ["行銷", "業務", "管理"],
        "employment_rate": 80, "overseas_ratio": 3,
        "industry_outlook": "負面",
        "ai_impact": "高度衝擊",
        "notes": "非頂尖商管出路競爭激烈，入學前需謹慎評估",
        "license_required": False,
        "double_major_friendly": False,
    },
]

# ============================================================
# 向量記憶（RAG）
# ============================================================

memory_store = []

def get_embedding(text):
    if not GEMINI_AVAILABLE:
        return np.random.rand(768)
    res = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(res["embedding"])

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return np.dot(a, b) / denom if denom else 0

def store_memory(text, tag="general"):
    vec = get_embedding(text)
    memory_store.append({
        "text": text, "vec": vec, "tag": tag,
        "time": datetime.now().isoformat()
    })
    # 記憶上限 200 筆，超過刪最舊
    if len(memory_store) > 200:
        memory_store.pop(0)

def retrieve_memory(query, top_k=3, tag=None):
    if not memory_store:
        return ""
    q_vec = get_embedding(query)
    pool = [m for m in memory_store if tag is None or m["tag"] == tag]
    scored = sorted([(cosine_sim(q_vec, m["vec"]), m["text"]) for m in pool], reverse=True)
    return "\n".join(t for _, t in scored[:top_k])

# ============================================================
# 分數計算 & 落點預測
# ============================================================

def calculate_score(scores: dict, weights: dict) -> float:
    return sum(scores.get(s, 0) * weights[s] for s in weights)

def predict_cutoff(m: dict) -> float:
    base = m["last_year_score"]
    base += {"上升": +2, "持平": 0, "下降": -2}.get(m["trend"], 0)
    base += {"高": +1, "中": 0, "低": -1}.get(m["stability"], 0)
    base += {"正面": +1, "穩定": 0, "負面": -2}.get(m.get("industry_outlook", "穩定"), 0)
    return round(base, 1)

def match_majors(scores: dict, min_gap: float = -5) -> list:
    results = []
    for m in majors_db:
        student_score = calculate_score(scores, m["weights"])
        cutoff = predict_cutoff(m)
        gap = student_score - cutoff
        if gap >= min_gap:
            safety = "穩上" if gap >= 3 else ("邊緣" if gap >= 0 else "衝刺")
            results.append({
                "school": m["school"],
                "major": m["major"],
                "group": m["group"],
                "student_score": round(student_score, 1),
                "cutoff": cutoff,
                "gap": round(gap, 1),
                "safety": safety,
                "salary_avg": m.get("salary_avg", 0),
                "employment_rate": m.get("employment_rate", 0),
                "career": m["career"],
                "industry_outlook": m.get("industry_outlook", "穩定"),
                "ai_impact": m.get("ai_impact", "未知"),
                "notes": m.get("notes", ""),
                "overseas_ratio": m.get("overseas_ratio", 0),
                "license_required": m.get("license_required", False),
                "double_major_friendly": m.get("double_major_friendly", False),
            })
    results.sort(key=lambda x: x["gap"], reverse=True)
    return results

# ============================================================
# Markdown → HTML 轉換
# ============================================================

def md_to_html(text: str) -> str:
    text = re.sub(r"^#### (.*)$", r"<h4>\1</h4>", text, flags=re.M)
    text = re.sub(r"^### (.*)$",  r"<h3>\1</h3>",  text, flags=re.M)
    text = re.sub(r"^## (.*)$",   r"<h2>\1</h2>",   text, flags=re.M)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*",     r"<em>\1</em>",         text)
    text = re.sub(r"^[-•] (.*)$",   r"<li>\1</li>",          text, flags=re.M)
    text = re.sub(r"(<li>.*?</li>\n?)+", lambda m: f"<ul>{m.group()}</ul>", text, flags=re.S)
    text = re.sub(r"\n{2,}", "</p><p>", text)
    return f"<p>{text}</p>"

# ============================================================
# AI 分析建議（Gemini 2.5 Flash）
# ============================================================

def generate_advice(profile: dict, matches: list) -> str:
    # 快取：同樣的 profile + 前5筆 matches 就回傳快取
    cache_key = make_cache_key(profile, [m["major"] for m in matches[:5]])
    cached = cache_get(cache_key)
    if cached:
        return cached

    context = retrieve_memory(str(profile))
    abroad = profile.get("出國意願", "n")
    abroad_text = (
        "請提供國外升學策略（美國/英國/日本/新加坡），包含推薦研究所、準備方向、TOEFL/GRE門檻"
        if abroad == "y"
        else "出國升學部分略過"
    )
    top_matches = matches[:5]

    prompt = f"""
你是台灣頂尖升學顧問，有15年實戰經驗，精通學測落點分析。

【學生基本資料】
稱呼：{profile.get("name", "同學")}
學測各科分數：{json.dumps(profile.get("scores", {}), ensure_ascii=False)}
興趣領域：{profile.get("interests", "未填")}
擅長科目：{profile.get("strengths", "未填")}
個人特質：{profile.get("personality", "未填")}
出國意願：{"有" if abroad == "y" else "無"}
優先考量：{profile.get("偏好", "未填")}

【落點分析結果（前5名）】
{json.dumps(top_matches, ensure_ascii=False, indent=2)}

【歷史諮詢參考】
{context if context else "（無相關歷史紀錄）"}

【2025 產業時事】
- AI工具普及衝擊：基層程式員、翻譯、文字工作需求明顯下降
- 半導體持續缺工：台積電、聯發科、英業達持續擴編，台灣製造業護城河
- 護理缺工嚴重：政策薪資補貼，月薪已逼近 7-8 萬
- AI人才需求爆增：機器學習、MLOps、資料科學職缺同比翻倍
- 非頂尖商管科系出路持續惡化，需靠雙主修或輔系突圍

請提供以下分析（務實、直接、像顧問在面對面諮詢，不要空話）：

**1. 志願推薦序（最多4個，明確標出穩上/目標/衝刺）**
每個志願說明：推薦原因 + 風險評估 + 與學生特質契合度

**2. 能力匹配警告**
哪些科系只有興趣符合但能力不足（直接說，不要客氣）

**3. 英文影響力分析**
英文加權對這位學生的具體影響（加幾分？差幾個科系？）

**4. 薪資與出路務實評估**
各推薦科系：畢業即薪 / 3年後 / 10年後的薪資預期
AI時代下的風險與機會

**5. {abroad_text}**

**6. 雙主修 / 輔系建議**
根據落點，哪些雙主修組合能讓這位學生大幅提升競爭力

**7. 給這位同學的一句話**
語氣：務實、直接、像朋友不是業務員
格式：條列式，重點加粗
"""

    if not GEMINI_AVAILABLE:
        return "<p>⚠️ AI 分析模組未載入，請確認 GEMINI_API_KEY 設定</p>"

    model = genai.GenerativeModel(MODEL_NAME)
    res = model.generate_content(prompt)
    html = md_to_html(res.text)

    cache_set(cache_key, html)
    store_memory(
        f"學生：{profile} 推薦：{[m['major'] for m in top_matches[:3]]}",
        tag="consultation"
    )
    return html

# ============================================================
# 聊天歷史（多輪對話）
# ============================================================

chat_sessions: dict[str, list] = {}

def get_or_create_session(session_id: str) -> list:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = [
            {
                "role": "user",
                "parts": ["""你是台灣升學諮詢顧問，有 15 年實戰經驗。
請用繁體中文回答，語氣像朋友而非業務員，回答務實不說廢話，每次回答不超過 350 字，善用條列式。
你熟知：學測落點、各科系出路、薪資行情、AI 時代產業趨勢、備審資料撰寫、面試技巧。"""]
            },
            {"role": "model", "parts": ["好的，我準備好了！有任何升學問題都可以問我。"]}
        ]
    return chat_sessions[session_id]

# ============================================================
# API 路由
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "升志 ScoreWise API v3.0",
        "model": MODEL_NAME,
        "time": datetime.now().isoformat()
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json or {}
        scores = data.get("scores", {})
        profile = data.get("profile", {})
        profile["scores"] = scores

        for s in ["國文", "英文", "數學"]:
            if s not in scores:
                return jsonify({"status": "error", "message": f"缺少科目分數：{s}"}), 400

        matches = match_majors(scores, min_gap=-5)

        if not matches:
            return jsonify({
                "status": "success",
                "result": "<p>目前分數較難錄取已知科系，建議補考或調整方向。</p>",
                "matches": [],
                "summary": {"total": 0, "safe": 0, "borderline": 0, "reach": 0}
            })

        summary = {
            "total": len(matches),
            "safe":       sum(1 for m in matches if m["safety"] == "穩上"),
            "borderline": sum(1 for m in matches if m["safety"] == "邊緣"),
            "reach":      sum(1 for m in matches if m["safety"] == "衝刺"),
        }

        ai_result = generate_advice(profile, matches)

        return jsonify({
            "status": "success",
            "result": ai_result,
            "matches": matches,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """多輪對話（帶歷史記憶）"""
    try:
        body = request.json or {}
        user_message = body.get("message", "").strip()
        session_id   = body.get("session_id", "default")

        if not user_message:
            return jsonify({"status": "error", "message": "訊息不能為空"}), 400
        if not GEMINI_AVAILABLE:
            return jsonify({"status": "error", "message": "AI 模組未啟用"}), 503

        # 快取：同一個問題 1 小時內回傳快取
        cache_key = make_cache_key("chat", user_message)
        cached = cache_get(cache_key)
        if cached and session_id == "default":
            return jsonify({"status": "ok", "reply": cached, "cached": True})

        history = get_or_create_session(session_id)
        context = retrieve_memory(user_message, top_k=2, tag="consultation")

        full_message = user_message
        if context:
            full_message += f"\n\n（參考背景：{context}）"

        history.append({"role": "user", "parts": [full_message]})

        model = genai.GenerativeModel(MODEL_NAME)
        chat_obj = model.start_chat(history=history[:-1])
        res = chat_obj.send_message(full_message)
        reply = res.text

        history.append({"role": "model", "parts": [reply]})

        # 超過 20 輪對話刪掉最早的（保留系統 prompt 前兩筆）
        if len(history) > 22:
            history[2:4] = []

        store_memory(f"Q:{user_message} A:{reply[:120]}", tag="chat")
        cache_set(cache_key, reply)

        return jsonify({"status": "ok", "reply": reply})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/majors", methods=["GET"])
def get_majors():
    group  = request.args.get("group")
    school = request.args.get("school")
    data   = majors_db
    if group:
        data = [m for m in data if m.get("group") == group]
    if school:
        data = [m for m in data if m.get("school") == school]
    # 不回傳向量欄位
    safe = [{k: v for k, v in m.items() if k != "vec"} for m in data]
    return jsonify({"status": "ok", "data": safe, "count": len(safe)})


@app.route("/predict/<school>/<major>", methods=["GET"])
def predict_single(school, major):
    target = next((m for m in majors_db if m["school"] == school and m["major"] == major), None)
    if not target:
        return jsonify({"status": "error", "message": "找不到該科系"}), 404
    cutoff = predict_cutoff(target)
    return jsonify({
        "status": "ok",
        "school": school,
        "major": major,
        "last_year": target["last_year_score"],
        "predicted_cutoff": cutoff,
        "industry_outlook": target.get("industry_outlook"),
        "ai_impact": target.get("ai_impact"),
        "notes": target.get("notes", "")
    })


@app.route("/compare", methods=["POST"])
def compare_majors():
    """比較多個科系"""
    try:
        body   = request.json or {}
        scores = body.get("scores", {})
        majors = body.get("majors", [])  # [{"school":"..","major":".."}]

        results = []
        for req in majors:
            target = next(
                (m for m in majors_db
                 if m["school"] == req.get("school") and m["major"] == req.get("major")),
                None
            )
            if not target:
                continue
            student_score = calculate_score(scores, target["weights"]) if scores else None
            cutoff = predict_cutoff(target)
            results.append({
                **{k: v for k, v in target.items() if k not in ("weights",)},
                "predicted_cutoff": cutoff,
                "student_score": student_score,
                "gap": round(student_score - cutoff, 1) if student_score is not None else None,
            })

        return jsonify({"status": "ok", "comparison": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/salary-insight", methods=["GET"])
def salary_insight():
    """薪資排行榜"""
    group = request.args.get("group")
    data  = majors_db if not group else [m for m in majors_db if m.get("group") == group]
    ranked = sorted(data, key=lambda m: m.get("salary_avg", 0), reverse=True)
    result = [
        {
            "school": m["school"], "major": m["major"],
            "salary_avg": m.get("salary_avg", 0),
            "employment_rate": m.get("employment_rate", 0),
            "industry_outlook": m.get("industry_outlook"),
            "ai_impact": m.get("ai_impact"),
        }
        for m in ranked
    ]
    return jsonify({"status": "ok", "data": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "gemini": GEMINI_AVAILABLE,
        "memory_size": len(memory_store),
        "cache_size": len(_cache),
        "majors_count": len(majors_db),
        "active_sessions": len(chat_sessions),
    })


# ============================================================
# 啟動
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)