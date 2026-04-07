import os, re, json, time, hashlib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# 初始化
# ============================================================

# ⚠️  唯一需要改的地方：模型名稱
# gemini-1.5-flash 已下架 → 改用 gemini-2.0-flash
MODEL_NAME = "gemini-2.0-flash"
GEMINI_AVAILABLE = False
genai = None

try:
    import google.generativeai as genai_module
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        genai_module.configure(api_key=api_key)
        # 做一次快速驗證，若 key 無效會拋例外
        _ = genai_module.GenerativeModel(MODEL_NAME)
        genai = genai_module
        GEMINI_AVAILABLE = True
        print(f"[OK] Gemini {MODEL_NAME} 初始化成功")
    else:
        print("[WARN] 未偵測到 GEMINI_API_KEY，AI 功能停用")
except ImportError:
    print("[WARN] google-generativeai 套件未安裝")
except Exception as e:
    print(f"[ERROR] Gemini 初始化失敗：{e}")

app = Flask(__name__)
CORS(app, origins="*")   # 允許 GitHub Pages 跨域

# ============================================================
# 科系資料庫（從 majors.json 讀取；若不存在則使用內建預設）
# ============================================================

def load_majors() -> list:
    """優先讀 majors.json，讀不到才用內建資料。"""
    try:
        with open("majors.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] 讀入 majors.json，共 {len(data)} 筆科系")
        return data
    except FileNotFoundError:
        print("[INFO] majors.json 不存在，使用內建科系資料")
    except Exception as e:
        print(f"[WARN] majors.json 讀取失敗：{e}，使用內建科系資料")

    # 內建預設（最小化，實際部署請使用 majors.json）
    return [
        {
            "school": "台灣大學", "major": "電機工程", "group": "理工",
            "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
            "thresholds": {"數學": 12, "自然": 11},
            "last_year_score": 58, "trend": "上升", "stability": "高",
            "last_year_cutoff_by_subject": {"數學": 14, "自然": 13, "英文": 12, "國文": 10},
            "salary_avg": 85000, "career": ["半導體", "IC設計", "AI工程"],
            "employment_rate": 97, "overseas_ratio": 35,
            "industry_outlook": "正面", "ai_impact": "受益",
            "notes": "台積電、聯發科主要來源校系",
            "license_required": False, "double_major_friendly": True,
        },
        {
            "school": "台灣大學", "major": "資訊工程", "group": "理工",
            "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
            "thresholds": {"數學": 13},
            "last_year_score": 60, "trend": "上升", "stability": "高",
            "last_year_cutoff_by_subject": {"數學": 15, "英文": 13, "自然": 12, "國文": 10},
            "salary_avg": 90000, "career": ["AI", "軟體工程", "新創"],
            "employment_rate": 98, "overseas_ratio": 40,
            "industry_outlook": "正面", "ai_impact": "受益",
            "notes": "矽谷就業率最高的台灣科系之一",
            "license_required": False, "double_major_friendly": True,
        },
        {
            "school": "清華大學", "major": "資訊工程", "group": "理工",
            "weights": {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
            "thresholds": {"數學": 12},
            "last_year_score": 56, "trend": "上升", "stability": "高",
            "last_year_cutoff_by_subject": {"數學": 13, "英文": 12, "自然": 11, "國文": 9},
            "salary_avg": 82000, "career": ["AI", "軟體", "半導體"],
            "employment_rate": 96, "overseas_ratio": 30,
            "industry_outlook": "正面", "ai_impact": "受益",
            "notes": "竹科地緣優勢明顯",
            "license_required": False, "double_major_friendly": True,
        },
        {
            "school": "成功大學", "major": "醫學系", "group": "醫",
            "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
            "thresholds": {"自然": 13, "英文": 12, "數學": 11},
            "last_year_score": 65, "trend": "持平", "stability": "高",
            "last_year_cutoff_by_subject": {"自然": 14, "英文": 13, "數學": 12, "國文": 11},
            "salary_avg": 120000, "career": ["醫師", "研究醫學"],
            "employment_rate": 99, "overseas_ratio": 5,
            "industry_outlook": "穩定", "ai_impact": "部分影響",
            "notes": "南台灣最頂尖醫學系",
            "license_required": True, "double_major_friendly": False,
        },
        {
            "school": "台灣大學", "major": "法律系", "group": "社會",
            "weights": {"國文": 2, "英文": 2, "數學": 1, "自然": 0, "社會": 3},
            "thresholds": {"社會": 11, "國文": 10},
            "last_year_score": 57, "trend": "持平", "stability": "中",
            "last_year_cutoff_by_subject": {"社會": 13, "英文": 12, "國文": 11, "數學": 9},
            "salary_avg": 70000, "career": ["律師", "法務", "公職"],
            "employment_rate": 88, "overseas_ratio": 15,
            "industry_outlook": "穩定", "ai_impact": "部分影響",
            "notes": "需考律師證照才能發揮最大效益",
            "license_required": True, "double_major_friendly": True,
        },
        {
            "school": "台灣大學", "major": "護理系", "group": "醫",
            "weights": {"國文": 1, "英文": 1, "數學": 1, "自然": 2, "社會": 1},
            "thresholds": {"自然": 8},
            "last_year_score": 48, "trend": "上升", "stability": "高",
            "last_year_cutoff_by_subject": {"自然": 10, "英文": 9, "國文": 8, "數學": 7},
            "salary_avg": 65000, "career": ["護理師", "護理長", "醫療管理"],
            "employment_rate": 98, "overseas_ratio": 10,
            "industry_outlook": "正面", "ai_impact": "低影響",
            "notes": "護理缺工嚴重，薪資近年大幅提升",
            "license_required": True, "double_major_friendly": False,
        },
        {
            "school": "台灣大學", "major": "外文系", "group": "人文",
            "weights": {"國文": 2, "英文": 3, "數學": 0, "自然": 0, "社會": 1},
            "thresholds": {"英文": 12},
            "last_year_score": 50, "trend": "下降", "stability": "低",
            "last_year_cutoff_by_subject": {"英文": 13, "國文": 11, "社會": 10},
            "salary_avg": 42000, "career": ["翻譯", "外貿", "教師"],
            "employment_rate": 75, "overseas_ratio": 20,
            "industry_outlook": "負面", "ai_impact": "高度衝擊",
            "notes": "AI翻譯工具衝擊傳統翻譯市場",
            "license_required": False, "double_major_friendly": True,
        },
        {
            "school": "政治大學", "major": "財務管理", "group": "社會",
            "weights": {"國文": 1, "英文": 2, "數學": 2, "自然": 0, "社會": 2},
            "thresholds": {"數學": 9},
            "last_year_score": 50, "trend": "持平", "stability": "中",
            "last_year_cutoff_by_subject": {"數學": 11, "英文": 11, "社會": 10, "國文": 8},
            "salary_avg": 65000, "career": ["投資銀行", "基金", "企業財務"],
            "employment_rate": 89, "overseas_ratio": 20,
            "industry_outlook": "穩定", "ai_impact": "部分影響",
            "notes": "政大商院品牌效益強",
            "license_required": False, "double_major_friendly": True,
        },
    ]

majors_db = load_majors()

# ============================================================
# 快取層（節省 API 費用）
# ============================================================

_cache: dict = {}
CACHE_TTL = 3600

def cache_get(key):
    item = _cache.get(key)
    if item and time.time() - item["ts"] < CACHE_TTL:
        return item["val"]
    return None

def cache_set(key, val):
    _cache[key] = {"val": val, "ts": time.time()}
    # 超過 500 筆就清掉最舊的 100 筆
    if len(_cache) > 500:
        oldest = sorted(_cache, key=lambda k: _cache[k]["ts"])[:100]
        for k in oldest:
            del _cache[k]

def make_cache_key(*args):
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()

# ============================================================
# 向量記憶（RAG）
# ============================================================

memory_store: list = []
MAX_MEMORY = 300

def get_embedding(text: str) -> np.ndarray:
    """Gemini text-embedding-004（可用則呼叫，否則用 hash 向量）。"""
    if GEMINI_AVAILABLE and genai:
        try:
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(res["embedding"])
        except Exception:
            pass
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
    return np.random.default_rng(seed).random(768)

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

def store_memory(text: str, tag: str = "general"):
    vec = get_embedding(text)
    memory_store.append({"text": text, "vec": vec, "tag": tag, "time": datetime.now().isoformat()})
    if len(memory_store) > MAX_MEMORY:
        memory_store.pop(0)

def retrieve_memory(query: str, top_k: int = 3, tag: str = None) -> str:
    if not memory_store:
        return ""
    q_vec = get_embedding(query)
    pool = [m for m in memory_store if tag is None or m["tag"] == tag]
    if not pool:
        return ""
    scored = sorted([(cosine_sim(q_vec, m["vec"]), m["text"]) for m in pool], reverse=True)
    return "\n".join(t for _, t in scored[:top_k])

# 初始化知識庫
store_memory(
    "2025產業趨勢：AI工具衝擊基層程式員、翻譯、文字工作。"
    "半導體持續缺工（台積電、聯發科）。護理薪資提升至7-8萬。"
    "AI人才需求暴增（ML、MLOps、資料科學）。非頂尖商管出路持續惡化。",
    tag="knowledge"
)

# ============================================================
# PR 值計算
# ============================================================

_PR_TABLE = {
    15:99, 14:96, 13:90, 12:82, 11:72,
    10:60,  9:48,  8:36,  7:25,  6:16,
     5: 9,  4: 5,  3: 2,  2: 1,  1: 0
}

def calculate_pr(score: int) -> int:
    return _PR_TABLE.get(max(1, min(15, int(score))), 0)

def calculate_combined_pr(scores: dict) -> int:
    if not scores:
        return 0
    return round(sum(calculate_pr(v) for v in scores.values()) / len(scores))

# ============================================================
# 分數計算 & 落點預測
# ============================================================

def calculate_score(scores: dict, weights: dict) -> float:
    return sum(scores.get(s, 0) * weights.get(s, 0) for s in weights)

def predict_cutoff(m: dict) -> float:
    base = m.get("last_year_score", 50)
    base += {"上升": +2, "持平": 0, "下降": -2}.get(m.get("trend", "持平"), 0)
    base += {"高": +1, "中": 0, "低": -1}.get(m.get("stability", "中"), 0)
    base += {"正面": +1, "穩定": 0, "負面": -2}.get(m.get("industry_outlook", "穩定"), 0)
    return round(base, 1)

def match_majors(scores: dict, min_gap: int = -3) -> list:
    """以「倍率篩選決勝科目」邏輯比對學測級分。"""
    results = []
    for m in majors_db:
        weights = m.get("weights", {})
        thresholds = m.get("thresholds", {})
        cutoff_by_subject = m.get("last_year_cutoff_by_subject", {})
        active = {s: w for s, w in weights.items() if w > 0}
        if not active:
            continue

        # STEP 1：門檻過濾
        fail_threshold = False
        for subj, req in thresholds.items():
            if scores.get(subj, 0) < req:
                fail_threshold = True
                break
        if fail_threshold:
            continue

        # STEP 2：決勝科目 gap
        min_mult = min(active.values())
        tiebreak_subjects = [s for s, w in active.items() if w == min_mult]
        gap = None
        tiebreak_subject = None
        for subj in tiebreak_subjects:
            student_val = scores.get(subj, 0)
            cutoff_val = cutoff_by_subject.get(subj)
            if cutoff_val is None:
                continue
            g = student_val - cutoff_val
            if gap is None or g < gap:
                gap = g
                tiebreak_subject = subj

        if gap is None or gap < min_gap:
            continue

        # STEP 3：安全分類
        if gap >= 2:
            safety = "穩上"
        elif gap >= -1:
            safety = "目標"
        else:
            safety = "衝刺"

        subject_prs = {s: calculate_pr(scores.get(s, 0)) for s in active}
        combined_pr = calculate_combined_pr({s: scores.get(s, 0) for s in active})

        results.append({
            "school": m["school"],
            "major": m["major"],
            "group": m.get("group", ""),
            "tiebreak_subject": tiebreak_subject,
            "tiebreak_multiplier": min_mult,
            "student_tiebreak_score": scores.get(tiebreak_subject, 0),
            "cutoff_tiebreak_score": cutoff_by_subject.get(tiebreak_subject),
            "gap": gap,
            "safety": safety,
            "weights": active,
            "salary_avg": m.get("salary_avg", 0),
            "employment_rate": m.get("employment_rate", 0),
            "career": m.get("career", []),
            "industry_outlook": m.get("industry_outlook", "穩定"),
            "ai_impact": m.get("ai_impact", "未知"),
            "notes": m.get("notes", ""),
            "overseas_ratio": m.get("overseas_ratio", 0),
            "license_required": m.get("license_required", False),
            "double_major_friendly": m.get("double_major_friendly", False),
            "combined_pr": combined_pr,
            "subject_prs": subject_prs,
        })

    results.sort(key=lambda x: x["gap"], reverse=True)
    return results

# ============================================================
# Markdown → HTML
# ============================================================

def md_to_html(text: str) -> str:
    text = re.sub(r"^#### (.*)$", r"<h4>\1</h4>", text, flags=re.M)
    text = re.sub(r"^### (.*)$",  r"<h3>\1</h3>",  text, flags=re.M)
    text = re.sub(r"^## (.*)$",   r"<h2>\1</h2>",   text, flags=re.M)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    lines = text.split("\n")
    out, in_list = [], False
    for line in lines:
        m2 = re.match(r"^[-•] (.*)$", line)
        if m2:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"  <li>{m2.group(1)}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(line)
    if in_list:
        out.append("</ul>")
    text = "\n".join(out)
    text = re.sub(r"\n{2,}", "</p><p>", text.strip())
    return f"<p>{text}</p>"

# ============================================================
# AI 分析（Gemini 2.0 Flash）
# ============================================================

def generate_advice(profile: dict, matches: list) -> str:
    """呼叫 Gemini 生成升學建議報告。快取相同輸入避免重複計費。"""
    cache_key = make_cache_key("advice", profile.get("scores"), [m["major"] for m in matches[:5]])
    cached = cache_get(cache_key)
    if cached:
        return cached

    context = (
        retrieve_memory(str(profile.get("interests", "")), top_k=2, tag="consultation")
        + "\n"
        + retrieve_memory("產業趨勢", tag="knowledge")
    )

    abroad = profile.get("出國意願", "n")
    abroad_section = (
        "5. 出國升學策略（美國/英國/日本/新加坡）：推薦研究所、TOEFL/GRE門檻、準備方向"
        if abroad == "y"
        else "5. 出國升學：此同學目前無意願，略過"
    )

    # 精簡 matches，只送前 5 筆給 AI（節省 token）
    compact = [
        {
            "校系": f"{m['school']} {m['major']}",
            "安全": m["safety"],
            "決勝科": m.get("tiebreak_subject", ""),
            "差距": m["gap"],
            "月薪均": m.get("salary_avg", 0),
            "就業率": m.get("employment_rate", 0),
            "AI衝擊": m.get("ai_impact", ""),
            "可雙主修": m.get("double_major_friendly", False),
        }
        for m in matches[:5]
    ]

    scores_str = json.dumps(profile.get("scores", {}), ensure_ascii=False)
    prompt = f"""你是台灣頂尖升學顧問（15年實戰，務實直接，像朋友不像業務）。

學生資料：
- 稱呼：{profile.get('name', '同學')}
- 學測成績（15級分制）：{scores_str}
- 擅長科目：{profile.get('strengths', '未填')}
- 興趣領域：{profile.get('interests', '未填')}
- 個人特質：{profile.get('personality', '未填')}
- 出國意願：{'有' if abroad == 'y' else '無'}
- 優先考量：{profile.get('偏好', '未填')}

落點結果（前5名）：
{json.dumps(compact, ensure_ascii=False, indent=2)}

產業背景（2025）：
{context}

請輸出以下七點分析（條列式，重點加粗，禁止廢話）：

**1. 志願推薦序（最多4個）**
- 標明 穩上/目標/衝刺 + 推薦原因 + 與學生特質契合度

**2. 能力不匹配警告**
- 哪些只有興趣符合但能力不足（直說，不客氣）

**3. 英文影響力分析**
- 英文加權對這位學生的具體影響（差幾分？差幾個科系？）

**4. 薪資與出路務實評估**
- 各推薦科系：畢業起薪 / 3年後 / 10年後
- AI時代下的風險與機會

{abroad_section}

**6. 雙主修／輔系建議**
- 哪些組合能大幅提升競爭力

**7. 給這位同學的一句話**
"""

    if not GEMINI_AVAILABLE or not genai:
        return "<p>⚠️ AI 分析模組未啟用。請至 Render Dashboard → Environment → 新增環境變數 <code>GEMINI_API_KEY</code>，填入你的 Google AI Studio API Key。</p>"

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        res = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1200, "temperature": 0.7}
        )
        html = md_to_html(res.text)
        cache_set(cache_key, html)
        store_memory(
            f"學生{profile.get('name', '')} 推薦：{[m['major'] for m in matches[:3]]}",
            tag="consultation"
        )
        return html
    except Exception as e:
        error_msg = str(e)
        # 友善的錯誤提示
        if "API_KEY" in error_msg.upper() or "400" in error_msg:
            return (
                "<p>⚠️ Gemini API Key 無效。請確認：<br>"
                "1. Render Dashboard → Environment Variables → GEMINI_API_KEY 是否已設定<br>"
                "2. Key 是否來自 <a href='https://aistudio.google.com/app/apikey'>Google AI Studio</a><br>"
                "3. Key 前後是否有多餘空格</p>"
            )
        return f"<p>⚠️ AI 分析暫時失敗（{error_msg[:120]}），其他落點資料仍有效。</p>"

# ============================================================
# 多輪對話 Session（帶 TTL 自動清理）
# ============================================================

SESSION_TTL = 3600
chat_sessions: dict = {}

SYSTEM_PROMPT = (
    "你是台灣升學諮詢顧問，有15年實戰經驗。"
    "請用繁體中文回答，語氣像朋友而非業務員，"
    "每次回答不超過350字，善用條列式，務實不說廢話。"
    "你熟知：學測落點、各科系出路、薪資行情、AI時代產業趨勢、備審撰寫、面試技巧。"
)

def get_or_create_session(session_id: str) -> list:
    now = time.time()
    expired = [sid for sid, s in chat_sessions.items() if now - s["ts"] > SESSION_TTL]
    for sid in expired:
        del chat_sessions[sid]

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": [
                {"role": "user", "parts": [SYSTEM_PROMPT]},
                {"role": "model", "parts": ["好的！有任何升學問題都可以問我。"]}
            ],
            "ts": now,
        }
    chat_sessions[session_id]["ts"] = now
    return chat_sessions[session_id]["history"]

# ============================================================
# API 路由
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "升志 ScoreWise API v3.2",
        "model": MODEL_NAME,
        "gemini_ready": GEMINI_AVAILABLE,
        "majors_count": len(majors_db),
        "time": datetime.now().isoformat()
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json or {}
        scores = data.get("scores", {})
        profile = data.get("profile", {})
        profile["scores"] = scores

        # 必填驗證
        for s in ["國文", "英文", "數學"]:
            if s not in scores:
                return jsonify({"status": "error", "message": f"缺少科目分數：{s}"}), 400

        # 分數範圍驗證（學測 1~15 級分）
        for subj, val in scores.items():
            try:
                val = int(val)
            except (TypeError, ValueError):
                return jsonify({"status": "error", "message": f"{subj} 分數格式錯誤"}), 400
            if not (1 <= val <= 15):
                return jsonify({"status": "error", "message": f"{subj} 需為 1~15 的學測級分"}), 400
            scores[subj] = val  # 確保是整數

        matches = match_majors(scores, min_gap=-3)

        if not matches:
            return jsonify({
                "status": "success",
                "result": "<p>目前分數較難符合已知科系門檻，建議補考或調整填寫科目（例如：是否有填社會或數B？）。</p>",
                "matches": [],
                "summary": {"total": 0, "safe": 0, "target": 0, "reach": 0}
            })

        summary = {
            "total":  len(matches),
            "safe":   sum(1 for m in matches if m["safety"] == "穩上"),
            "target": sum(1 for m in matches if m["safety"] == "目標"),
            "reach":  sum(1 for m in matches if m["safety"] == "衝刺"),
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
    try:
        body = request.json or {}
        user_message = body.get("message", "").strip()
        session_id   = body.get("session_id", "default")

        if not user_message:
            return jsonify({"status": "error", "message": "訊息不能為空"}), 400

        if not GEMINI_AVAILABLE or not genai:
            return jsonify({
                "status": "error",
                "reply": "AI 模組未啟用，請設定 GEMINI_API_KEY 環境變數。"
            }), 503

        # 快取（同 session + 同訊息 1 小時內不重複呼叫）
        cache_key = make_cache_key("chat", session_id, user_message)
        cached = cache_get(cache_key)
        if cached:
            return jsonify({"status": "ok", "reply": cached, "cached": True})

        history = get_or_create_session(session_id)

        # 補充 RAG 背景
        context = retrieve_memory(user_message, top_k=2, tag="consultation")
        msg_with_ctx = user_message + (f"\n\n（參考背景：{context}）" if context else "")

        history.append({"role": "user", "parts": [msg_with_ctx]})

        model = genai.GenerativeModel(MODEL_NAME)
        chat_obj = model.start_chat(history=history[:-1])
        res = chat_obj.send_message(
            msg_with_ctx,
            generation_config={"max_output_tokens": 500, "temperature": 0.7}
        )
        reply = res.text

        history.append({"role": "model", "parts": [reply]})

        # 保留最近 20 輪（系統 prompt 2 則不動）
        if len(history) > 22:
            history[2:4] = []

        store_memory(f"Q:{user_message[:80]} A:{reply[:100]}", tag="chat")
        cache_set(cache_key, reply)

        return jsonify({"status": "ok", "reply": reply})

    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper() or "400" in error_msg:
            return jsonify({
                "status": "error",
                "reply": "API Key 無效，請至 Render Dashboard 確認 GEMINI_API_KEY 設定。"
            }), 401
        return jsonify({"status": "error", "message": error_msg}), 500


@app.route("/majors", methods=["GET"])
def get_majors():
    group  = request.args.get("group")
    school = request.args.get("school")
    data   = majors_db
    if group:
        data = [m for m in data if m.get("group") == group]
    if school:
        data = [m for m in data if m.get("school") == school]
    safe = [{k: v for k, v in m.items() if k != "vec"} for m in data]
    return jsonify({"status": "ok", "data": safe, "count": len(safe)})


@app.route("/predict/<school>/<major>", methods=["GET"])
def predict_single(school, major):
    target = next((m for m in majors_db if m["school"] == school and m["major"] == major), None)
    if not target:
        return jsonify({"status": "error", "message": "找不到該科系"}), 404
    return jsonify({
        "status": "ok",
        "school": school,
        "major": major,
        "last_year": target.get("last_year_score"),
        "predicted_cutoff": predict_cutoff(target),
        "industry_outlook": target.get("industry_outlook"),
        "ai_impact": target.get("ai_impact"),
        "notes": target.get("notes", "")
    })


@app.route("/compare", methods=["POST"])
def compare_majors():
    try:
        body   = request.json or {}
        scores = body.get("scores", {})
        majors = body.get("majors", [])
        results = []
        for req in majors:
            target = next(
                (m for m in majors_db if m["school"] == req.get("school") and m["major"] == req.get("major")),
                None
            )
            if not target:
                continue
            cutoff = predict_cutoff(target)
            student_score = calculate_score(scores, target.get("weights", {})) if scores else None
            results.append({
                **{k: v for k, v in target.items() if k not in ("vec",)},
                "predicted_cutoff": cutoff,
                "student_score": round(student_score, 1) if student_score is not None else None,
                "gap": round(student_score - cutoff, 1) if student_score is not None else None,
            })
        return jsonify({"status": "ok", "comparison": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/salary-insight", methods=["GET"])
def salary_insight():
    group  = request.args.get("group")
    data   = majors_db if not group else [m for m in majors_db if m.get("group") == group]
    ranked = sorted(data, key=lambda m: m.get("salary_avg", 0), reverse=True)
    return jsonify({
        "status": "ok",
        "data": [
            {
                "school": m["school"], "major": m["major"],
                "salary_avg": m.get("salary_avg", 0),
                "employment_rate": m.get("employment_rate", 0),
                "industry_outlook": m.get("industry_outlook"),
                "ai_impact": m.get("ai_impact"),
            }
            for m in ranked
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "gemini_available": GEMINI_AVAILABLE,
        "memory_size": len(memory_store),
        "cache_size": len(_cache),
        "majors_count": len(majors_db),
        "active_sessions": len(chat_sessions),
        "api_key_set": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
    })


# ============================================================
# 啟動
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
