import os, re, json, time, hashlib
import numpy as np
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# ============================================================
# 初始化
# ============================================================

MODEL_NAME = "gemini-2.0-flash"
GEMINI_AVAILABLE = False
genai = None

try:
    import google.generativeai as genai_module
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        genai_module.configure(api_key=api_key)
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
CORS(app, origins="*")

# ============================================================
# 科目名稱對照
# majors.json 使用「數學A」「數學B」；前端送來也用這組名稱
# ============================================================

ALL_SUBJECTS = ["國文", "英文", "數學A", "數學B", "自然", "社會", "物理", "化學", "生物", "地科"]

# 學校分類（用於偏好排序）
TOP_SCHOOLS = {"國立臺灣大學", "國立清華大學", "國立交通大學", "國立陽明交通大學", "國立成功大學"}
NORTH_KEYWORDS = ["臺北", "台北", "基隆", "新北", "桃園", "新竹", "宜蘭"]
SOUTH_KEYWORDS = ["台南", "臺南", "高雄", "屏東", "嘉義", "台東", "臺東", "澎湖"]

LOG_FILE = "user_queries.jsonl"

def log_query(scores: dict, school_pref: str) -> None:
    """將查詢記錄寫入 JSONL，不儲存任何個人識別資訊。"""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),  # UTC 時間
        "school_pref": school_pref,
        "scores": scores,          # 分數本身不含個資
        # ❌ 不記錄：IP、user-agent、session id、姓名等
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        logging.warning(f"[log_query] 寫入日誌失敗：{e}")  # 日誌失敗不應中斷主流程


def school_region(school: str) -> str:
    for kw in NORTH_KEYWORDS:
        if kw in school:
            return "north"
    for kw in SOUTH_KEYWORDS:
        if kw in school:
            return "south"
    return "central"

# ============================================================
# 科系資料庫（從 majors.json 讀取）
# ============================================================

def load_majors() -> list:
    try:
        with open("majors.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] 讀入 majors.json，共 {len(data)} 筆科系")
        return data
    except FileNotFoundError:
        print("[WARN] majors.json 不存在，使用內建最小資料集")
    except Exception as e:
        print(f"[WARN] majors.json 讀取失敗：{e}")

    # 內建最小資料集（格式與 majors.json 一致）
    return [
        {
            "school": "國立臺灣大學", "major": "電機工程學系", "group": "理工",
            "salary_median": "150萬",
            "thresholds": {"數學A": 14, "自然": 13},
            "multipliers": {"數學A": 2, "自然": 2, "英文": 1, "國文": 1},
            "last_year_cutoff_by_subject": {"數學A": 15, "自然": 14, "英文": 13, "國文": 11},
        },
        {
            "school": "國立臺灣大學", "major": "資訊工程學系", "group": "理工",
            "salary_median": "160萬",
            "thresholds": {"數學A": 15, "英文": 13},
            "multipliers": {"數學A": 2, "英文": 2, "自然": 1, "國文": 1},
            "last_year_cutoff_by_subject": {"數學A": 15, "英文": 14, "自然": 13, "國文": 11},
        },
        {
            "school": "國立成功大學", "major": "醫學系", "group": "醫藥",
            "salary_median": "250萬+",
            "thresholds": {"國文": 15, "英文": 15, "數學A": 15, "自然": 15},
            "multipliers": {"國文": 1, "英文": 1, "數學A": 1, "自然": 1},
            "last_year_cutoff_by_subject": {"自然": 15, "數學A": 15, "英文": 15, "國文": 15},
        },
    ]

majors_db = load_majors()

# ============================================================
# 薪資字串 → 數字（萬元/年）
# ============================================================

def parse_salary_median(raw) -> int:
    """把 '150萬'、'250萬+' 等字串轉成年薪整數（萬元）。"""
    if isinstance(raw, (int, float)):
        return int(raw)
    if not raw:
        return 0
    s = str(raw).replace("+", "").replace(",", "").replace("，", "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if m:
        return int(float(m.group(1)))
    return 0

# ============================================================
# 快取層
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
    15: 99, 14: 96, 13: 90, 12: 82, 11: 72,
    10: 60,  9: 48,  8: 36,  7: 25,  6: 16,
     5:  9,  4:  5,  3:  2,  2:  1,  1:  0
}

def calculate_pr(score: int) -> int:
    return _PR_TABLE.get(max(1, min(15, int(score))), 0)

def calculate_combined_pr(scores: dict) -> int:
    if not scores:
        return 0
    return round(sum(calculate_pr(v) for v in scores.values()) / len(scores))

# ============================================================
# 落點配對（讀 multipliers，科目名對應 majors.json 格式）
# ============================================================

def match_majors(scores: dict, min_gap: int = -3) -> list:
    """
    scores 的 key 格式與 majors.json 一致：數學A、數學B、國文…等。

    修改說明（v2）：
    - 移除所有 continue 跳過邏輯（類組、學校、分數門檻皆不直接排除）。
    - 改以 is_preferred 標記「完全符合」者（所有門檻皆達標）。
    - status 四分法：安全(>=2)、目標(0~2)、挑戰(-2~0)、困難(<-2)。
    - 回傳結果優先依 is_preferred 置頂，次依 diff 降冪排序。
    - 回傳 JSON 包含 is_preferred、diff 與更新後的 status 欄位。
    """
    results = []

    for m in majors_db:
        multipliers: dict = m.get("multipliers", {})
        thresholds: dict  = m.get("thresholds", {})
        cutoff_map: dict  = m.get("last_year_cutoff_by_subject", {})

        # 過濾出學生有分數且倍率 > 0 的科目
        active = {s: w for s, w in multipliers.items() if w > 0 and s in scores}
        if not active:
            # 學生完全沒有該科系所需科目的成績 → 仍納入，標記為非偏好
            # 無法計算決勝差值，gap 設為 None，略過此筆
            continue

        # ── is_preferred 判斷：門檻全部達標才為 True ──
        # 學生未填寫的科目視為 0，門檻不達標則 is_preferred = False（但不 continue）
        passes_threshold = all(
            scores.get(subj, 0) >= req
            for subj, req in thresholds.items()
        )
        is_preferred = passes_threshold

        # ── 決勝科目計算（最低倍率科目中，取差值最小者）──
        min_mult = min(active.values())
        tiebreak_subjects = [s for s, w in active.items() if w == min_mult]

        gap = None
        tiebreak_subject = None
        for subj in tiebreak_subjects:
            student_val = scores.get(subj, 0)
            cutoff_val  = cutoff_map.get(subj)
            if cutoff_val is None:
                continue
            g = student_val - cutoff_val
            if gap is None or g < gap:
                gap = g
                tiebreak_subject = subj

        # 無法取得決勝差值 → 略過
        if gap is None:
            continue

        # 差距過大（超出顯示意義）的衝刺志願才略過
        if gap < min_gap:
            continue

        # ── diff：決勝科目差值（與 gap 相同，語義更明確）──
        diff = gap

        # ── status 四分法 ──
        if diff >= 2:
            status = "安全"
        elif diff >= 0:
            status = "目標"
        elif diff >= -2:
            status = "挑戰"
        else:
            status = "困難"

        # ── safety（前端沿用的三分法標籤，維持相容）──
        if diff >= 2:
            safety = "穩上"
        elif diff >= -1:
            safety = "目標"
        else:
            safety = "衝刺"

        # 各科分數詳情（供前端安全區間視覺化）
        subject_detail = {}
        for subj in multipliers:
            if multipliers[subj] > 0:
                student_val = scores.get(subj, 0)
                cutoff_val  = cutoff_map.get(subj)
                subject_detail[subj] = {
                    "student":     student_val,
                    "cutoff":      cutoff_val,
                    "gap":         (student_val - cutoff_val) if cutoff_val is not None else None,
                    "multiplier":  multipliers[subj],
                    "is_tiebreak": (subj == tiebreak_subject),
                }

        # 薪資解析（保留原始 raw 字串，月薪估算備用）
        salary_year    = parse_salary_median(m.get("salary_median", 0))
        salary_monthly = round(salary_year * 10000 / 12) if salary_year else 0

        combined_pr = calculate_combined_pr({s: scores.get(s, 0) for s in active})

        results.append({
            # ── 核心欄位 ──
            "school":                  m["school"],
            "major":                   m["major"],
            "group":                   m.get("group", ""),
            "is_preferred":            is_preferred,          # ★ 完全符合門檻旗標
            "diff":                    diff,                   # ★ 決勝科目差值
            "status":                  status,                 # ★ 四分法狀態
            "safety":                  safety,                 # 前端三分法相容欄位
            # ── 決勝科目詳情 ──
            "tiebreak_subject":        tiebreak_subject,
            "tiebreak_multiplier":     min_mult,
            "student_tiebreak_score":  scores.get(tiebreak_subject, 0) if tiebreak_subject else 0,
            "cutoff_tiebreak_score":   cutoff_map.get(tiebreak_subject) if tiebreak_subject else None,
            "gap":                     gap,                    # 與 diff 相同，保留相容
            "subject_detail":          subject_detail,
            "multipliers":             multipliers,
            # ── 薪資 ──
            "salary_year_wan":         salary_year,
            "salary_monthly":          salary_monthly,
            "salary_median_raw":       m.get("salary_median", "—"),
            # ── 其他資訊 ──
            "career":                  m.get("career", []),
            "industry_outlook":        m.get("industry_outlook", "穩定"),
            "ai_impact":               m.get("ai_impact", "未知"),
            "notes":                   m.get("notes", ""),
            "overseas_ratio":          m.get("overseas_ratio", 0),
            "license_required":        m.get("license_required", False),
            "double_major_friendly":   m.get("double_major_friendly", True),
            "employment_rate":         m.get("employment_rate", 0),
            "combined_pr":             combined_pr,
            "school_is_national":      "國立" in m["school"],
            "school_region":           school_region(m["school"]),
        })

    # ── 排序：is_preferred 置頂（True=0, False=1），次依 diff 降冪 ──
    results.sort(key=lambda x: (0 if x["is_preferred"] else 1, -x["diff"]))
    return results


def sort_by_school_pref(matches: list, pref: str) -> list:
    """
    學校偏好二次排序（不過濾，不改變 is_preferred 置頂邏輯）。
    排序優先級：① is_preferred 置頂 → ② 符合學校偏好 → ③ diff 降冪
    """
    if pref == "any":
        # 僅依 is_preferred 置頂 + diff 降冪（match_majors 已做，直接回傳）
        return matches

    def sort_key(m: dict) -> tuple:
        school = m["school"]
        # is_preferred 置頂（True → 0, False → 1）
        pref_tier = 0 if m.get("is_preferred", False) else 1
        # 學校偏好符合（符合 → 0，不符合 → 1）
        if pref == "top3":
            school_tier = 0 if school in TOP_SCHOOLS else 1
        elif pref == "north":
            school_tier = 0 if m["school_region"] == "north" else 1
        elif pref == "south":
            school_tier = 0 if m["school_region"] == "south" else 1
        elif pref == "national":
            school_tier = 0 if m["school_is_national"] else 1
        else:
            school_tier = 0
        return (pref_tier, school_tier, -m["diff"])

    return sorted(matches, key=sort_key)

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
        "5. 出國升學策略（美國/英國/日本/新加坡）：推薦研究所、TOEFL/GRE門檻"
        if abroad == "y"
        else "5. 出國升學：此同學目前無意願，略過"
    )

    compact = [
        {
            "校系": f"{m['school']} {m['major']}",
            "安全": m["safety"],
            "決勝科": m.get("tiebreak_subject", ""),
            "差距": m["gap"],
            "年薪中位": m.get("salary_median_raw", "—"),
            "AI衝擊": m.get("ai_impact", "未知"),
            "可雙主修": m.get("double_major_friendly", True),
        }
        for m in matches[:5]
    ]

    scores_str = json.dumps(profile.get("scores", {}), ensure_ascii=False)
    prompt = f"""你是台灣具備產業洞察力的升學專家（務實直接，像朋友不像業務）。

學生資料：
- 稱呼：{profile.get('name', '同學')}
- 學測成績（15級分制）：{scores_str}
- 擅長科目：{profile.get('strengths', '未填')}
- 興趣領域：{profile.get('interests', '未填')}
- 個人特質：{profile.get('personality', '未填')}
- 出國意願：{'有' if abroad == 'y' else '無'}

落點結果（前5名）：
{json.dumps(compact, ensure_ascii=False, indent=2)}

產業背景（2025）：
{context}

請輸出以下分析（條列式，重點加粗，禁止廢話）：

**1. 志願推薦序（最多4個）**
- 標明 穩上/目標/衝刺 + 推薦原因 + 與學生特質契合度

**2. 能力不匹配警告**
- 哪些只有興趣符合但能力不足（直說，不客氣）

**3. 英文影響力分析**
- 英文加權對這位學生的具體影響

**4. 薪資與出路務實評估**
- 各推薦科系：畢業起薪 / 3年後 / 10年後
- AI時代下的風險與機會

{abroad_section}

**6. 雙主修／輔系建議**
- 哪些組合能大幅提升競爭力

**7. 給這位同學的一句話**
"""

    if not GEMINI_AVAILABLE or not genai:
        return (
            "<p>⚠️ AI 分析模組未啟用。請至 Render Dashboard → Environment → "
            "新增環境變數 <code>GEMINI_API_KEY</code>。</p>"
        )

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
        if "API_KEY" in error_msg.upper() or "400" in error_msg:
            return (
                "<p>⚠️ Gemini API Key 無效。請確認 Render Dashboard 中的 GEMINI_API_KEY 設定。</p>"
            )
        return f"<p>⚠️ AI 分析暫時失敗（{error_msg[:120]}），落點資料仍可參考。</p>"

# ============================================================
# 多輪對話 Session
# ============================================================

SESSION_TTL = 3600
chat_sessions: dict = {}

SYSTEM_PROMPT = (
    "你是台灣具備產業洞察力的升學專家。"
    "請用繁體中文回答，語氣要專業且具備前瞻性。"
    "每次回答不超過350字，善用條列式，務實不說廢話。"
    "分析學生的優勢學科適合哪些未來職涯；結合當前 AI、半導體或永續轉型等時事，分析該科系的前景；根據分數與門檻的 diff，給予具體的『加強科目建議』或『面試準備方向』。"
)

def get_or_create_session(session_id: str) -> list:
    now = time.time()
    expired = [sid for sid, s in chat_sessions.items() if now - s["ts"] > SESSION_TTL]
    for sid in expired:
        del chat_sessions[sid]
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": [
                {"role": "user",  "parts": [SYSTEM_PROMPT]},
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
        "message": "升志 ScoreWise API v4.0",
        "model": MODEL_NAME,
        "gemini_ready": GEMINI_AVAILABLE,
        "majors_count": len(majors_db),
        "time": datetime.now().isoformat()
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data    = request.json or {}
        scores  = data.get("scores", {})
        profile = data.get("profile", {})

        # ── Step 0：科目別名正規化（在任何驗證之前先統一 key）──
        scores = normalize_subject_keys(scores)

        # ── Step 1：必填驗證（國文、英文必須存在且值 >= 1）──
        # 用 scores.get(s, 0) == 0 同時捕捉「key 不存在」和「key 存在但值為 0」兩種情況
        missing = [s for s in ["國文", "英文"] if scores.get(s, 0) == 0]
        if missing:
            return jsonify({"status": "error", "message": f"缺少必填科目分數：{', '.join(missing)}（需為 1~15 級分）"}), 400

        # ── Step 2：log 原始輸入（在補 0 之前，記錄真實值）──
        school_pref = profile.get("school_pref", "any")
        log_query(scores, school_pref)

        # ── Step 3：選考科目補 0（容錯，內部使用）──
        for subj in ["數學A", "數學B", "自然", "社會"]:
            if subj not in scores:
                scores[subj] = 0

        # ── Step 4：分數範圍驗證（1~15；0 代表未選考，允許通過）──
        for subj, val in list(scores.items()):
            try:
                val = int(val)
            except (TypeError, ValueError):
                return jsonify({"status": "error", "message": f"{subj} 分數格式錯誤"}), 400
            if val != 0 and not (1 <= val <= 15):   # 0 不觸發錯誤
                return jsonify({"status": "error", "message": f"{subj} 需為 1~15 的學測級分（收到：{val}）"}), 400
            scores[subj] = val

        # ── Step 5：profile scores 在 int 轉換後設定，確保值為 int ──
        profile["scores"] = scores

        # ── Step 6：落點配對 + 排序（各只呼叫一次）──
        matches = match_majors(scores, min_gap=-3)
        matches = sort_by_school_pref(matches, school_pref)

        if not matches:
            return jsonify({
                "status": "success",
                "result": (
                    "<p>目前分數條件下，資料庫中的科系門檻較難符合。"
                    "建議確認選考科目是否正確填寫（數學A/B、自然、社會），"
                    "或嘗試放寬到更多科系。</p>"
                ),
                "matches": [],
                "summary": {"total": 0, "safe": 0, "target": 0, "challenge": 0, "hard": 0}
            })

        # summary 使用 status 四分法計數
        summary = {
            "total":     len(matches),
            "safe":      sum(1 for m in matches if m["status"] == "安全"),
            "target":    sum(1 for m in matches if m["status"] == "目標"),
            "challenge": sum(1 for m in matches if m["status"] == "挑戰"),
            "hard":      sum(1 for m in matches if m["status"] == "困難"),
            # 前端三分法相容欄位
            "穩上":      sum(1 for m in matches if m["safety"] == "穩上"),
            "目標":      sum(1 for m in matches if m["safety"] == "目標"),
            "衝刺":      sum(1 for m in matches if m["safety"] == "衝刺"),
        }

        ai_result = generate_advice(profile, matches)

        return jsonify({
            "status":  "success",
            "result":  ai_result,
            "matches": matches,
            "summary": summary
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        body         = request.json or {}
        user_message = body.get("message", "").strip()
        session_id   = body.get("session_id", "default")

        if not user_message:
            return jsonify({"status": "error", "message": "訊息不能為空"}), 400
        if not GEMINI_AVAILABLE or not genai:
            return jsonify({
                "status": "error",
                "reply": "AI 模組未啟用，請設定 GEMINI_API_KEY 環境變數。"
            }), 503

        cache_key = make_cache_key("chat", session_id, user_message)
        cached = cache_get(cache_key)
        if cached:
            return jsonify({"status": "ok", "reply": cached, "cached": True})

        history = get_or_create_session(session_id)
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
                "reply": "API Key 無效，請確認 GEMINI_API_KEY 設定。"
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
    return jsonify({"status": "ok", "data": data, "count": len(data)})


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
