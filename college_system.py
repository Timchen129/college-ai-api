import os
import re
import json
import time
import hashlib
import numpy as np
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
 
# === 初始化 ===
load_dotenv()
 
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
 
app = Flask(__name__)
CORS(app)
 
# ============================================================
# 資料層：科系資料庫（未來可替換成 DB / API）
# ============================================================
 
majors_db = [
    {
        "school": "台灣大學", "major": "電機工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "last_year_score": 58, "trend": "上升", "stability": "高",
        "required_abilities": ["數學", "物理"], "bonus_factor": ["英文"],
        "salary": "高", "salary_avg": 85000,
        "career": ["半導體", "IC設計", "AI工程"],
        "employment_rate": 97, "overseas_ratio": 35,
        "industry_outlook": "正面",  # 由 AI 分析更新
        "notes": "台積電、聯發科主要來源校系"
    },
    {
        "school": "台灣大學", "major": "資訊工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "last_year_score": 60, "trend": "上升", "stability": "高",
        "required_abilities": ["數學", "邏輯"], "bonus_factor": ["英文"],
        "salary": "高", "salary_avg": 90000,
        "career": ["AI", "軟體工程", "新創"],
        "employment_rate": 98, "overseas_ratio": 40,
        "industry_outlook": "正面",
        "notes": "矽谷就業率最高的台灣科系之一"
    },
    {
        "school": "清華大學", "major": "資訊工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "last_year_score": 56, "trend": "上升", "stability": "高",
        "required_abilities": ["數學", "邏輯"], "bonus_factor": ["英文"],
        "salary": "高", "salary_avg": 82000,
        "career": ["AI", "軟體", "半導體"],
        "employment_rate": 96, "overseas_ratio": 30,
        "industry_outlook": "正面",
        "notes": "竹科地緣優勢明顯"
    },
    {
        "school": "清華大學", "major": "電機工程", "group": "理工",
        "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "last_year_score": 55, "trend": "持平", "stability": "高",
        "required_abilities": ["數學", "物理"], "bonus_factor": [],
        "salary": "高", "salary_avg": 80000,
        "career": ["半導體", "IC設計", "電力電子"],
        "employment_rate": 96, "overseas_ratio": 28,
        "industry_outlook": "正面",
        "notes": ""
    },
    {
        "school": "成功大學", "major": "醫學系", "group": "醫",
        "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
        "last_year_score": 65, "trend": "持平", "stability": "高",
        "required_abilities": ["記憶力", "自然"], "bonus_factor": [],
        "salary": "高", "salary_avg": 120000,
        "career": ["醫師", "研究醫學"],
        "employment_rate": 99, "overseas_ratio": 5,
        "industry_outlook": "穩定",
        "notes": "南台灣最頂尖醫學系，錄取分數極高"
    },
    {
        "school": "台灣大學", "major": "法律系", "group": "社會",
        "weights": {"國文": 2, "英文": 2, "數學": 1, "自然": 0, "社會": 3},
        "last_year_score": 57, "trend": "持平", "stability": "中",
        "required_abilities": ["閱讀", "邏輯"], "bonus_factor": ["國文"],
        "salary": "中高", "salary_avg": 70000,
        "career": ["律師", "法務", "公職"],
        "employment_rate": 88, "overseas_ratio": 15,
        "industry_outlook": "穩定",
        "notes": "需考律師證照才能發揮最大效益"
    },
    {
        "school": "台灣大學", "major": "經濟系", "group": "社會",
        "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 0, "社會": 2},
        "last_year_score": 55, "trend": "上升", "stability": "中",
        "required_abilities": ["數學", "分析"], "bonus_factor": ["英文"],
        "salary": "中高", "salary_avg": 72000,
        "career": ["金融", "顧問", "政策研究"],
        "employment_rate": 90, "overseas_ratio": 25,
        "industry_outlook": "穩定",
        "notes": "出國深造比例高"
    },
    {
        "school": "陽明交大", "major": "醫學系", "group": "醫",
        "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
        "last_year_score": 67, "trend": "持平", "stability": "高",
        "required_abilities": ["記憶力", "自然"], "bonus_factor": [],
        "salary": "高", "salary_avg": 125000,
        "career": ["醫師", "研究醫學"],
        "employment_rate": 99, "overseas_ratio": 8,
        "industry_outlook": "穩定",
        "notes": "與台大醫學系並列最頂尖"
    },
    {
        "school": "私立大學", "major": "資訊工程", "group": "理工",
        "weights": {"國文": 1, "英文": 1, "數學": 2, "自然": 1, "社會": 0},
        "last_year_score": 40, "trend": "下降", "stability": "低",
        "required_abilities": ["數學"], "bonus_factor": [],
        "salary": "中", "salary_avg": 55000,
        "career": ["軟體開發", "IT"],
        "employment_rate": 82, "overseas_ratio": 5,
        "industry_outlook": "負面",  # AI時代影響基層程式員需求
        "notes": "AI工具普及後，非頂尖資工系就業壓力增加"
    },
    {
        "school": "私立大學", "major": "商業管理", "group": "社會",
        "weights": {"國文": 2, "英文": 2, "數學": 1, "自然": 0, "社會": 2},
        "last_year_score": 42, "trend": "下降", "stability": "低",
        "required_abilities": ["溝通"], "bonus_factor": [],
        "salary": "中", "salary_avg": 45000,
        "career": ["行銷", "業務", "管理"],
        "employment_rate": 80, "overseas_ratio": 3,
        "industry_outlook": "負面",
        "notes": "非頂尖商管出路競爭激烈"
    },
    {
        "school": "台灣大學", "major": "護理系", "group": "醫",
        "weights": {"國文": 1, "英文": 1, "數學": 1, "自然": 2, "社會": 1},
        "last_year_score": 48, "trend": "上升", "stability": "高",
        "required_abilities": ["自然", "耐心"], "bonus_factor": [],
        "salary": "中高", "salary_avg": 65000,
        "career": ["護理師", "護理長", "醫療管理"],
        "employment_rate": 98, "overseas_ratio": 10,
        "industry_outlook": "正面",  # 缺工持續
        "notes": "護理缺工嚴重，薪資近年大幅提升"
    },
    {
        "school": "台灣大學", "major": "外文系", "group": "人文",
        "weights": {"國文": 2, "英文": 3, "數學": 0, "自然": 0, "社會": 1},
        "last_year_score": 50, "trend": "下降", "stability": "低",
        "required_abilities": ["英文", "語言"], "bonus_factor": [],
        "salary": "中低", "salary_avg": 42000,
        "career": ["翻譯", "外貿", "教師"],
        "employment_rate": 75, "overseas_ratio": 20,
        "industry_outlook": "負面",  # AI翻譯衝擊
        "notes": "AI翻譯工具衝擊傳統翻譯市場"
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
    return np.array(res['embedding'])
 
def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    return np.dot(a, b) / denom
 
def store_memory(text, tag="general"):
    vec = get_embedding(text)
    memory_store.append({"text": text, "vec": vec, "tag": tag, "time": datetime.now().isoformat()})
 
def retrieve_memory(query, top_k=3, tag=None):
    if not memory_store:
        return ""
    q_vec = get_embedding(query)
    pool = [m for m in memory_store if tag is None or m["tag"] == tag]
    scored = [(cosine_sim(q_vec, m["vec"]), m["text"]) for m in pool]
    scored.sort(reverse=True)
    return "\n".join([t for _, t in scored[:top_k]])
 
# ============================================================
# 分數計算 & 落點預測
# ============================================================
 
def calculate_score(scores: dict, weights: dict) -> float:
    return sum(scores.get(s, 0) * weights[s] for s in weights)
 
def predict_cutoff(m: dict) -> float:
    """
    預測今年最低錄取分數
    考量因子：去年分數、趨勢、穩定度、產業前景、AI影響
    """
    base = m["last_year_score"]
 
    # 歷史趨勢
    trend_delta = {"上升": +2, "持平": 0, "下降": -2}
    base += trend_delta.get(m["trend"], 0)
 
    # 穩定性
    stability_delta = {"高": +1, "中": 0, "低": -1}
    base += stability_delta.get(m["stability"], 0)
 
    # 產業前景（AI時代調整）
    outlook_delta = {"正面": +1, "穩定": 0, "負面": -2}
    base += outlook_delta.get(m.get("industry_outlook", "穩定"), 0)
 
    return round(base, 1)
 
def match_majors(scores: dict, min_gap: float = -5) -> list:
    """
    min_gap: 允許低於預測錄取分幾分仍顯示（衝刺建議用）
    """
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
                "notes": m.get("notes", ""),
                "overseas_ratio": m.get("overseas_ratio", 0),
            })
 
    # 依照 gap 降冪排序
    results.sort(key=lambda x: x["gap"], reverse=True)
    return results
 
# ============================================================
# AI 分析建議（Gemini）
# ============================================================
 
def generate_advice(profile: dict, matches: list) -> str:
    context = retrieve_memory(str(profile))
 
    abroad = profile.get("出國意願", "n")
    abroad_text = "請務必提供國外升學（美國/英國/日本/新加坡）策略與具體建議" if abroad == "y" else "國外升學部分略過"
 
    # 取前5個最相關科系給 AI 分析
    top_matches = matches[:5]
 
    prompt = f"""
你是台灣頂尖升學顧問，有15年實戰經驗，精通學測落點分析。
 
【學生基本資料】
姓名/代稱：{profile.get('name', '同學')}
學測各科分數：{profile.get('scores', {})}
興趣領域：{profile.get('interests', '未填')}
擅長科目：{profile.get('strengths', '未填')}
個人特質：{profile.get('personality', '未填')}
出國意願：{'有' if abroad == 'y' else '無'}
 
【落點分析結果（已篩選）】
{json.dumps(top_matches, ensure_ascii=False, indent=2)}
 
【歷史諮詢參考】
{context if context else '（無相關歷史紀錄）'}
 
【當前時事背景（2024-2025）】
- AI工具普及衝擊：基層程式員、翻譯、文字工作需求下降
- 半導體產業持續缺工：台積電、聯發科、英業達持續擴編
- 護理缺工嚴重：政策補貼推動薪資提升至6-8萬
- AI人才需求暴增：機器學習、資料科學相關職缺翻倍
- 非頂尖商管科系出路持續惡化
 
請提供以下分析（務必具體、不空話、像真實顧問在面對面諮詢）：
 
**1. 精準推薦志願序（最多4個，附理由）**
- 包含：穩上志願、目標志願、衝刺志願
- 每個志願說明：為何推薦、風險評估、與學生特質的契合度
 
**2. 能力匹配警告**
- 哪些科系只有興趣符合但能力不足（要直接說）
- 邊緣案例的具體風險
 
**3. 英文是否為決定性因素**
- 具體說明英文加權對此學生的影響
 
**4. 未來出路務實評估**
- 各推薦科系的薪資預期（畢業即、3年後、10年後）
- AI時代下各科系的風險與機會
 
**5. {abroad_text}**
 
**6. 給這位同學的一句話建議**
 
語氣：務實、直接、像朋友不是業務員
格式：條列式，重點加粗
"""
 
    if not GEMINI_AVAILABLE:
        return "<p>⚠️ AI 分析模組未載入，請確認 GEMINI_API_KEY 設定</p>"
 
    model = genai.GenerativeModel('gemini-1.5-flash')
    res = model.generate_content(prompt)
    text = res.text
 
    # Markdown → HTML 轉換
    text = re.sub(r'^#### (.*$)', r'<h4>\1</h4>', text, flags=re.M)
    text = re.sub(r'^### (.*$)', r'<h3>\1</h3>', text, flags=re.M)
    text = re.sub(r'^## (.*$)', r'<h2>\1</h2>', text, flags=re.M)
    text = re.sub(r'^\*\*\*\*(.*?)\*\*\*\*$', r'<h4>\1</h4>', text, flags=re.M)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'^[-•] (.*$)', r'<li>\1</li>', text, flags=re.M)
    text = re.sub(r'(<li>.*</li>\n?)+', lambda m: f'<ul>{m.group()}</ul>', text)
    text = re.sub(r'\n{2,}', '</p><p>', text)
    text = f'<p>{text}</p>'
 
    # 儲存到記憶
    store_memory(f"學生：{profile} 推薦：{[m['major'] for m in top_matches[:3]]}", tag="consultation")
 
    return text
 
# ============================================================
# 爬蟲模組（可選擇性啟用）
# ============================================================
 
def fetch_ptt_data(keyword: str, limit: int = 10) -> list:
    """
    從 PTT Exam 版爬取相關討論
    實際部署時需遵守 PTT 使用規範
    """
    # 示意結構，實際需用 requests + BeautifulSoup
    return [
        {"source": "PTT Exam", "content": f"關於{keyword}的學長姐討論（示例）", "date": "2024-12-01"}
    ]
 
def fetch_trend_score_adjustment(major_name: str) -> float:
    """
    根據 Google Trends / 新聞輿情調整分數預測
    未來可串接真實 API
    """
    # 示意邏輯
    ai_impact_keywords = ["資訊工程", "電機", "AI", "半導體"]
    decline_keywords = ["翻譯", "外文", "商管", "傳播"]
 
    if any(k in major_name for k in ai_impact_keywords):
        return 1.5  # 熱門科系加分
    elif any(k in major_name for k in decline_keywords):
        return -1.5  # 衰退科系減分
    return 0
 
# ============================================================
# API 路由
# ============================================================
 
@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "AI升學系統 API v2.0", "time": datetime.now().isoformat()})
 
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        scores = data.get("scores", {})
        profile = data.get("profile", {})
        profile["scores"] = scores
 
        # 驗證輸入
        required_subjects = ["國文", "英文", "數學"]
        for s in required_subjects:
            if s not in scores:
                return jsonify({"status": "error", "message": f"缺少科目分數：{s}"}), 400
 
        # 落點計算（包含衝刺志願）
        matches = match_majors(scores, min_gap=-5)
 
        if not matches:
            return jsonify({
                "status": "success",
                "result": "<h3>分析結果</h3><p>目前分數較難錄取已知科系，建議補考或調整方向。</p>",
                "matches": [],
                "summary": {"total": 0, "safe": 0, "borderline": 0, "reach": 0}
            })
 
        # 統計
        summary = {
            "total": len(matches),
            "safe": len([m for m in matches if m["safety"] == "穩上"]),
            "borderline": len([m for m in matches if m["safety"] == "邊緣"]),
            "reach": len([m for m in matches if m["safety"] == "衝刺"]),
        }
 
        # AI 分析
        ai_result = generate_advice(profile, matches)
 
        return jsonify({
            "status": "success",
            "result": ai_result,
            "matches": matches,
            "summary": summary
        })
 
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
 
@app.route("/majors", methods=["GET"])
def get_majors():
    """取得所有科系基本資訊"""
    group = request.args.get("group")
    data = majors_db if not group else [m for m in majors_db if m.get("group") == group]
    return jsonify({"status": "ok", "data": data, "count": len(data)})
 
 
@app.route("/predict/<school>/<major>", methods=["GET"])
def predict_single(school, major):
    """查詢特定科系的預測錄取分數"""
    target = next((m for m in majors_db if m["school"] == school and m["major"] == major), None)
    if not target:
        return jsonify({"status": "error", "message": "找不到該科系"}), 404
 
    cutoff = predict_cutoff(target)
    adjustment = fetch_trend_score_adjustment(major)
 
    return jsonify({
        "status": "ok",
        "school": school,
        "major": major,
        "last_year": target["last_year_score"],
        "predicted_cutoff": cutoff + adjustment,
        "trend_adjustment": adjustment,
        "industry_outlook": target.get("industry_outlook"),
        "notes": target.get("notes", "")
    })
 
 
@app.route("/chat", methods=["POST"])
def chat():
    """獨立 AI 問答（不需分數）"""
    try:
        user_message = request.json.get("message", "")
        if not user_message:
            return jsonify({"status": "error", "message": "訊息不能為空"}), 400
 
        context = retrieve_memory(user_message, top_k=2)
 
        prompt = f"""你是台灣升學諮詢顧問，請回答以下問題。
參考背景：{context if context else '無'}
問題：{user_message}
請用繁體中文、條列式、務實角度回答，不超過300字。"""
 
        if not GEMINI_AVAILABLE:
            return jsonify({"status": "error", "message": "AI 模組未啟用"}), 503
 
        model = genai.GenerativeModel('gemini-1.5-flash')
        res = model.generate_content(prompt)
 
        store_memory(f"Q:{user_message} A:{res.text[:100]}", tag="chat")
 
        return jsonify({"status": "ok", "reply": res.text})
 
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "gemini": GEMINI_AVAILABLE,
        "memory_size": len(memory_store),
        "majors_count": len(majors_db)
    })
 
 
# ============================================================
# 啟動
# ============================================================
 
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)