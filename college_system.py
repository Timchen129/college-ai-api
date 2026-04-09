import os, re, json, time, hashlib, math
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

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
# 第一階段：科目名稱對照表 & 正規化（修復 NameError）
# ============================================================

# 前端或不同版本 JSON 可能送來的科目別名 → 統一名稱
SUBJECT_ALIASES: dict[str, str] = {
    "數學":   "數學A",
    "數甲":   "數學A",
    "數學甲": "數學A",
    "math_a": "數學A",
    "數乙":   "數學B",
    "數學乙": "數學B",
    "math_b": "數學B",
    "自然科": "自然",
    "社會科": "社會",
    "國語":   "國文",
    "chinese":"國文",
    "english":"英文",
}

ALL_SUBJECTS = ["國文", "英文", "數學A", "數學B", "自然", "社會", "物理", "化學", "生物", "地科"]

def normalize_subject_keys(scores: dict) -> dict:
    """
    把前端送來的 scores dict 中的科目名稱正規化。
    例如 {"數甲": 14} → {"數學A": 14}
    不認識的 key 直接保留（避免誤刪）。
    """
    out = {}
    for k, v in scores.items():
        canonical = SUBJECT_ALIASES.get(k, k)
        out[canonical] = v
    return out

# ============================================================
# 學校分類
# ============================================================

TOP_SCHOOLS = {"國立臺灣大學", "國立清華大學", "國立交通大學", "國立陽明交通大學", "國立成功大學"}
NORTH_KEYWORDS = ["臺北", "台北", "基隆", "新北", "桃園", "新竹", "宜蘭"]
SOUTH_KEYWORDS = ["台南", "臺南", "高雄", "屏東", "嘉義", "台東", "臺東", "澎湖"]

def school_region(school: str) -> str:
    for kw in NORTH_KEYWORDS:
        if kw in school: return "north"
    for kw in SOUTH_KEYWORDS:
        if kw in school: return "south"
    return "central"

# ============================================================
# 第二階段：今年考試環境背景（寫死常數）
# ============================================================

EXAM_CONTEXT_2025 = {
    "year": 2025,
    "description": "115學年度學測環境摘要",
    "key_trends": [
        "數學A難度明顯上升，頂標預估下修至13-14級分",
        "英文聽力新題型上路，整體考生英文表現分布更分散",
        "自然科生物題量增加，物化題目情境化明顯",
        "考生總人數約13.2萬，較去年減少約6千人（少子化效應持續）",
        "AI相關科系（資工、電機、資管）報名熱度連續三年上升",
        "醫學/牙醫系競爭仍極激烈，預測頂標需求維持15級分",
        "商管類科系在私立大學缺額壓力增加，整體分數線下修",
        "半導體/IC設計相關科系：台積電、聯發科持續大量徵才，就業前景強勁",
    ],
    "score_distribution_notes": {
        "數學A": "頂標估13，前標估11，均標估8",
        "英文":  "頂標估14，前標估12，均標估9",
        "國文":  "頂標估14，前標估12，均標估10",
        "自然":  "頂標估13，前標估11，均標估8",
    },
    "hot_industries": ["AI/機器學習", "半導體", "生醫科技", "網路資安", "綠能電動車"],
    "declining_fields": ["傳統文字媒體", "一般翻譯", "非頂大商管"],
}

EXAM_CONTEXT_STR = "\n".join([
    f"【{EXAM_CONTEXT_2025['description']}】",
    "關鍵趨勢：",
    *[f"  · {t}" for t in EXAM_CONTEXT_2025["key_trends"]],
    "分數分布：",
    *[f"  · {k}：{v}" for k, v in EXAM_CONTEXT_2025["score_distribution_notes"].items()],
    f"熱門產業：{', '.join(EXAM_CONTEXT_2025['hot_industries'])}",
    f"承壓領域：{', '.join(EXAM_CONTEXT_2025['declining_fields'])}",
])

# ============================================================
# 科系資料庫（擴充格式：支援 past_thresholds / quota / industry_tags）
# ============================================================

def load_majors() -> list:
    try:
        with open("majors.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] 讀入 majors.json，共 {len(data)} 筆科系")
        return data
    except FileNotFoundError:
        print("[WARN] majors.json 不存在，使用內建擴充資料集")
    except Exception as e:
        print(f"[WARN] majors.json 讀取失敗：{e}")

    # 內建資料集（含新欄位示範）
    return [
        {
            "school": "國立臺灣大學", "major": "電機工程學系", "group": "理工",
            "salary_median": "150萬",
            "quota": 120,
            "industry_tags": ["半導體", "AI", "IC設計"],
            "thresholds": {"數學A": 14, "自然": 13},
            "multipliers": {"數學A": 2, "自然": 2, "英文": 1, "國文": 1},
            "last_year_cutoff_by_subject": {"數學A": 15, "自然": 14, "英文": 13, "國文": 11},
            "past_thresholds": {
                "111": {"數學A": 15, "自然": 14},
                "112": {"數學A": 15, "自然": 14},
                "113": {"數學A": 14, "自然": 13},
            },
            "ai_impact": "高度受益",
            "employment_rate": 98,
            "career": ["IC設計工程師", "韌體工程師", "AI研究員"],
        },
        {
            "school": "國立臺灣大學", "major": "資訊工程學系", "group": "理工",
            "salary_median": "160萬",
            "quota": 90,
            "industry_tags": ["AI", "軟體", "雲端"],
            "thresholds": {"數學A": 15, "英文": 13},
            "multipliers": {"數學A": 2, "英文": 2, "自然": 1, "國文": 1},
            "last_year_cutoff_by_subject": {"數學A": 15, "英文": 14, "自然": 13, "國文": 11},
            "past_thresholds": {
                "111": {"數學A": 15, "英文": 14},
                "112": {"數學A": 15, "英文": 14},
                "113": {"數學A": 15, "英文": 13},
            },
            "ai_impact": "高度受益",
            "employment_rate": 99,
            "career": ["軟體工程師", "ML工程師", "技術創業"],
        },
        {
            "school": "國立成功大學", "major": "醫學系", "group": "醫藥",
            "salary_median": "250萬+",
            "quota": 60,
            "industry_tags": ["醫療", "生醫"],
            "thresholds": {"國文": 15, "英文": 15, "數學A": 15, "自然": 15},
            "multipliers": {"國文": 1, "英文": 1, "數學A": 1, "自然": 1},
            "last_year_cutoff_by_subject": {"自然": 15, "數學A": 15, "英文": 15, "國文": 15},
            "past_thresholds": {
                "111": {"數學A": 15, "自然": 15, "英文": 15, "國文": 15},
                "112": {"數學A": 15, "自然": 15, "英文": 15, "國文": 15},
                "113": {"數學A": 15, "自然": 15, "英文": 15, "國文": 15},
            },
            "ai_impact": "輔助工具化",
            "employment_rate": 100,
            "career": ["臨床醫師", "醫學研究", "醫療管理"],
        },
        {
            "school": "國立臺灣大學", "major": "財務金融學系", "group": "商管",
            "salary_median": "100萬",
            "quota": 80,
            "industry_tags": ["金融科技", "投資銀行"],
            "thresholds": {"英文": 13, "數學B": 12},
            "multipliers": {"英文": 2, "數學B": 2, "國文": 1, "社會": 1},
            "last_year_cutoff_by_subject": {"英文": 14, "數學B": 13, "國文": 12, "社會": 12},
            "past_thresholds": {
                "111": {"英文": 14, "數學B": 13},
                "112": {"英文": 14, "數學B": 13},
                "113": {"英文": 13, "數學B": 12},
            },
            "ai_impact": "部分衝擊",
            "employment_rate": 88,
            "career": ["投資分析師", "金融科技", "財務顧問"],
        },
        {
            "school": "國立清華大學", "major": "奈米工程與微系統學系", "group": "理工",
            "salary_median": "140萬",
            "quota": 50,
            "industry_tags": ["半導體", "材料", "奈米科技"],
            "thresholds": {"數學A": 13, "自然": 12},
            "multipliers": {"數學A": 2, "自然": 2, "英文": 1, "國文": 1},
            "last_year_cutoff_by_subject": {"數學A": 14, "自然": 13, "英文": 12, "國文": 10},
            "past_thresholds": {
                "111": {"數學A": 14, "自然": 13},
                "112": {"數學A": 14, "自然": 13},
                "113": {"數學A": 13, "自然": 12},
            },
            "ai_impact": "受益",
            "employment_rate": 95,
            "career": ["製程工程師", "材料研發", "台積電相關職缺"],
        },
    ]

majors_db = load_majors()

# ============================================================
# 薪資字串 → 數字
# ============================================================

def parse_salary_median(raw) -> int:
    if isinstance(raw, (int, float)): return int(raw)
    if not raw: return 0
    s = str(raw).replace("+", "").replace(",", "").replace("，", "")
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return int(float(m.group(1))) if m else 0

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
        for k in oldest: del _cache[k]

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
                content=text, task_type="retrieval_document"
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
    if len(memory_store) > MAX_MEMORY: memory_store.pop(0)

def retrieve_memory(query: str, top_k: int = 3, tag: str = None) -> str:
    if not memory_store: return ""
    q_vec = get_embedding(query)
    pool = [m for m in memory_store if tag is None or m["tag"] == tag]
    if not pool: return ""
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
    15:99, 14:96, 13:90, 12:82, 11:72,
    10:60,  9:48,  8:36,  7:25,  6:16,
     5: 9,  4: 5,  3: 2,  2: 1,  1: 0
}

def calculate_pr(score: int) -> int:
    return _PR_TABLE.get(max(1, min(15, int(score))), 0)

def calculate_combined_pr(scores: dict) -> int:
    if not scores: return 0
    return round(sum(calculate_pr(v) for v in scores.values()) / len(scores))

# ============================================================
# 第一階段：邏輯鬆綁的落點配對
# ============================================================

def compute_admission_probability(gap: int, passed_threshold: bool, quota: int = 50) -> int:
    """
    根據分差、門檻狀況與錄取名額估算「AI預測錄取率」（%）。
    這是啟發式估算，非精確統計模型。
    """
    if not passed_threshold:
        # 未達門檻：根據差距給出低機率
        base = max(0, 5 + gap * 3)   # gap 是負數，如 -2 → 5-6 = -1 → 0
        return min(base, 15)

    # 已達門檻：gap 越大越穩
    if gap >= 3:    return 95
    if gap >= 2:    return 88
    if gap >= 1:    return 78
    if gap == 0:    return 62
    if gap == -1:   return 45
    if gap == -2:   return 28
    return max(5, 15 + gap * 5)

def match_majors(scores: dict) -> list:
    """
    邏輯鬆綁版：不再硬性過濾任何科系。
    門檻未達 → is_preferred = False、safety = "挑戰"、opacity 提示前端淡化。
    gap 極低（< -5）才跳過（防止清單爆炸）。
    """
    results = []
    MIN_GAP_HARD = -5  # 只過濾差距極大的（純防爆用）

    for m in majors_db:
        multipliers: dict = m.get("multipliers", {})
        thresholds: dict  = m.get("thresholds", {})
        cutoff_map: dict  = m.get("last_year_cutoff_by_subject", {})
        past: dict        = m.get("past_thresholds", {})

        # 過濾出學生有分數且倍率 > 0 的科目
        active = {s: w for s, w in multipliers.items() if w > 0 and s in scores}
        if not active:
            continue

        # ── 門檻檢查（不過濾，只標記）──
        failed_thresholds = {
            subj: req
            for subj, req in thresholds.items()
            if scores.get(subj, 0) < req
        }
        passed_threshold = len(failed_thresholds) == 0

        # ── 決勝科目計算 ──
        min_mult = min(active.values())
        tiebreak_subjects = [s for s, w in active.items() if w == min_mult]

        gap = None
        tiebreak_subject = None
        for subj in tiebreak_subjects:
            student_val = scores.get(subj, 0)
            cutoff_val  = cutoff_map.get(subj)
            if cutoff_val is None: continue
            g = student_val - cutoff_val
            if gap is None or g < gap:
                gap = g
                tiebreak_subject = subj

        if gap is None: continue
        if gap < MIN_GAP_HARD: continue  # 只過濾極端情況

        # ── 安全分類 ──
        if not passed_threshold:
            safety = "挑戰"
        elif gap >= 2:
            safety = "穩上"
        elif gap >= -1:
            safety = "目標"
        else:
            safety = "衝刺"

        # ── 各科詳情 ──
        subject_detail = {}
        for subj, mult in multipliers.items():
            if mult > 0:
                sv  = scores.get(subj, 0)
                cv  = cutoff_map.get(subj)
                subject_detail[subj] = {
                    "student":    sv,
                    "cutoff":     cv,
                    "gap":        (sv - cv) if cv is not None else None,
                    "multiplier": mult,
                    "is_tiebreak": (subj == tiebreak_subject),
                    "threshold":  thresholds.get(subj),
                    "below_threshold": sv < thresholds.get(subj, 0) if subj in thresholds else False,
                }

        # ── 歷年數據摘要 ──
        history_summary = {}
        for yr, thr in sorted(past.items()):
            history_summary[yr] = thr

        # ── AI 預測錄取率 ──
        quota = m.get("quota", 50)
        admission_prob = compute_admission_probability(gap, passed_threshold, quota)

        salary_year = parse_salary_median(m.get("salary_median", 0))
        combined_pr = calculate_combined_pr({s: scores.get(s, 0) for s in active})

        results.append({
            "school":              m["school"],
            "major":               m["major"],
            "group":               m.get("group", ""),
            "tiebreak_subject":    tiebreak_subject,
            "tiebreak_multiplier": min_mult,
            "gap":                 gap,
            "safety":              safety,
            "is_preferred":        passed_threshold,   # False = 挑戰志願
            "failed_thresholds":   failed_thresholds,  # 未達門檻詳情
            "subject_detail":      subject_detail,
            "multipliers":         multipliers,
            # ── 第二階段新欄位 ──
            "past_thresholds":     history_summary,
            "quota":               quota,
            "industry_tags":       m.get("industry_tags", []),
            "admission_prob":      admission_prob,     # AI 預測錄取率
            # ── 薪資 ──
            "salary_median_raw":   m.get("salary_median", "—"),
            "salary_year_wan":     salary_year,
            # ── 其他 ──
            "career":              m.get("career", []),
            "ai_impact":           m.get("ai_impact", "未知"),
            "notes":               m.get("notes", ""),
            "overseas_ratio":      m.get("overseas_ratio", 0),
            "license_required":    m.get("license_required", False),
            "double_major_friendly": m.get("double_major_friendly", True),
            "employment_rate":     m.get("employment_rate", 0),
            "combined_pr":         combined_pr,
            "school_is_national":  "國立" in m["school"],
            "school_region":       school_region(m["school"]),
        })

    # 排序：passed_threshold 優先，然後 gap 降冪
    results.sort(key=lambda x: (0 if x["is_preferred"] else 1, -x["gap"]))
    return results


def sort_by_school_pref(matches: list, pref: str) -> list:
    if pref == "any": return matches

    def pref_score(m: dict) -> int:
        s = m["school"]
        if pref == "top3":     return 0 if s in TOP_SCHOOLS else 1
        if pref == "north":    return 0 if m["school_region"] == "north" else 1
        if pref == "south":    return 0 if m["school_region"] == "south" else 1
        if pref == "national": return 0 if m["school_is_national"] else 1
        return 0

    return sorted(matches, key=lambda m: (pref_score(m), 0 if m["is_preferred"] else 1, -m["gap"]))

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
            if not in_list: out.append("<ul>"); in_list = True
            out.append(f"  <li>{m2.group(1)}</li>")
        else:
            if in_list: out.append("</ul>"); in_list = False
            out.append(line)
    if in_list: out.append("</ul>")
    text = "\n".join(out)
    text = re.sub(r"\n{2,}", "</p><p>", text.strip())
    return f"<p>{text}</p>"

# ============================================================
# 第三階段：強化 AI Prompt（歷史數據 + 環境背景 + 預測機率）
# ============================================================

def generate_advice(profile: dict, matches: list) -> str:
    cache_key = make_cache_key("advice_v2", profile.get("scores"), [m["major"] for m in matches[:5]])
    cached = cache_get(cache_key)
    if cached: return cached

    context = (
        retrieve_memory(str(profile.get("interests", "")), top_k=2, tag="consultation")
        + "\n"
        + retrieve_memory("產業趨勢", tag="knowledge")
    )
    abroad = profile.get("出國意願", "n")
    abroad_section = (
        "**5. 出國升學策略（美國/英國/日本/新加坡）**：推薦研究所、TOEFL/GRE門檻"
        if abroad == "y"
        else "5. 出國升學：此同學目前無意願，略過。"
    )

    # 前5名完整數據（含歷史 + 錄取名額 + 預測機率）
    compact = []
    for m in matches[:5]:
        entry = {
            "校系":       f"{m['school']} {m['major']}",
            "安全分類":   m["safety"],
            "門檻通過":   m["is_preferred"],
            "決勝科目差距": m["gap"],
            "AI預測錄取率": f"{m.get('admission_prob', 0)}%",
            "錄取名額":   m.get("quota", "未知"),
            "年薪中位":   m.get("salary_median_raw", "—"),
            "AI衝擊":     m.get("ai_impact", "未知"),
            "產業標籤":   m.get("industry_tags", []),
            "歷年門檻趨勢": m.get("past_thresholds", {}),
        }
        if m.get("failed_thresholds"):
            entry["⚠️未達門檻科目"] = m["failed_thresholds"]
        compact.append(entry)

    scores_str = json.dumps(profile.get("scores", {}), ensure_ascii=False)
    prompt = f"""你是台灣頂尖升學顧問（15年實戰，資料分析師思維，務實直接，像聰明的學長姐而非業務員）。
你手中有完整數據，請以「數據說話」的方式給出分析，避免空洞建議。

═══════════════════════
學生資料
═══════════════════════
- 稱呼：{profile.get('name', '同學')}
- 學測成績（15級分制）：{scores_str}
- 擅長科目：{profile.get('strengths', '未填')}
- 興趣領域：{profile.get('interests', '未填')}
- 個人特質：{profile.get('personality', '未填')}
- 出國意願：{'有' if abroad == 'y' else '無'}

═══════════════════════
落點數據（前5名，含歷年趨勢）
═══════════════════════
{json.dumps(compact, ensure_ascii=False, indent=2)}

═══════════════════════
115學年度考試環境背景
═══════════════════════
{EXAM_CONTEXT_STR}

═══════════════════════
諮詢記憶背景
═══════════════════════
{context}

═══════════════════════
請輸出以下分析（條列式，重點加粗，總字數不超過700字）
═══════════════════════

**1. 志願推薦序（最多4個）**
- 每個志願標明：[穩上/目標/衝刺/挑戰] + AI預測錄取率 + 推薦原因 + 與學生特質契合度
- 若為「挑戰」志願（未達門檻），須明確說明差多少、值不值得拼

**2. 歷年趨勢解讀**
- 根據111-113年數據，今年分數線預測會升還是降？給出明確判斷

**3. 環境背景影響**
- 今年數A偏難對這位學生有利還是不利？考生減少對錄取線的影響？

**4. 薪資與產業出路**
- 各推薦科系：起薪估算 / 產業前景 / AI時代風險
- 哪個選擇的「性價比」最高？直說

{abroad_section}

**6. 給這位同學的一句話**（要有針對性，不要廢話）
"""

    if not GEMINI_AVAILABLE or not genai:
        return (
            "<p>⚠️ AI 分析模組未啟用。請至 Render Dashboard → Environment → "
            "新增環境變數 <code>GEMINI_API_KEY</code>。</p>"
            "<p>落點卡片數據仍可參考。</p>"
        )

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        res = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 1400, "temperature": 0.65}
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
            return "<p>⚠️ Gemini API Key 無效，請確認 GEMINI_API_KEY 設定。</p>"
        return f"<p>⚠️ AI 分析暫時失敗（{error_msg[:120]}），落點卡片仍可參考。</p>"

# ============================================================
# 多輪對話
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
    for sid in expired: del chat_sessions[sid]
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": [
                {"role": "user",  "parts": [SYSTEM_PROMPT]},
                {"role": "model", "parts": ["好的！有任何升學問題都可以問我。"]},
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
        "message": "升志 ScoreWise API v5.0",
        "model": MODEL_NAME,
        "gemini_ready": GEMINI_AVAILABLE,
        "majors_count": len(majors_db),
        "exam_year": EXAM_CONTEXT_2025["year"],
        "time": datetime.now().isoformat(),
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data    = request.json or {}
        scores  = data.get("scores", {})
        profile = data.get("profile", {})

        # 正規化科目名稱（修復 NameError + 相容舊前端）
        scores = normalize_subject_keys(scores)
        profile["scores"] = scores

        # 必填驗證
        missing = [s for s in ["國文", "英文"] if s not in scores]
        if missing:
            return jsonify({"status": "error", "message": f"缺少科目分數：{', '.join(missing)}"}), 400
        if "數學A" not in scores and "數學B" not in scores:
            return jsonify({"status": "error", "message": "請至少填入 數學A 或 數學B"}), 400

        # 分數範圍驗證
        for subj, val in list(scores.items()):
            try:
                val = int(val)
            except (TypeError, ValueError):
                return jsonify({"status": "error", "message": f"{subj} 分數格式錯誤"}), 400
            if not (1 <= val <= 15):
                return jsonify({"status": "error", "message": f"{subj} 需為 1~15 的學測級分"}), 400
            scores[subj] = val

        matches = match_majors(scores)

        school_pref = profile.get("school_pref", "any")
        matches = sort_by_school_pref(matches, school_pref)

        if not matches:
            return jsonify({
                "status": "success",
                "result": (
                    "<p>目前分數條件下，資料庫中的科系無法比對。"
                    "請確認選考科目是否正確填寫（數學A/B、自然、社會）。</p>"
                ),
                "matches": [],
                "summary": {"total": 0, "safe": 0, "target": 0, "reach": 0, "challenge": 0},
            })

        summary = {
            "total":     len(matches),
            "safe":      sum(1 for m in matches if m["safety"] == "穩上"),
            "target":    sum(1 for m in matches if m["safety"] == "目標"),
            "reach":     sum(1 for m in matches if m["safety"] == "衝刺"),
            "challenge": sum(1 for m in matches if m["safety"] == "挑戰"),
        }

        ai_result = generate_advice(profile, matches)

        return jsonify({
            "status":  "success",
            "result":  ai_result,
            "matches": matches,
            "summary": summary,
            "exam_context": EXAM_CONTEXT_2025["key_trends"][:3],  # 前端用
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
            return jsonify({"status": "error", "reply": "AI 模組未啟用，請設定 GEMINI_API_KEY。"}), 503

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
        if len(history) > 22: history[2:4] = []

        store_memory(f"Q:{user_message[:80]} A:{reply[:100]}", tag="chat")
        cache_set(cache_key, reply)
        return jsonify({"status": "ok", "reply": reply})

    except Exception as e:
        error_msg = str(e)
        if "API_KEY" in error_msg.upper() or "400" in error_msg:
            return jsonify({"status": "error", "reply": "API Key 無效，請確認設定。"}), 401
        return jsonify({"status": "error", "message": error_msg}), 500


@app.route("/majors", methods=["GET"])
def get_majors():
    group  = request.args.get("group")
    school = request.args.get("school")
    data   = majors_db
    if group:  data = [m for m in data if m.get("group") == group]
    if school: data = [m for m in data if m.get("school") == school]
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
        "exam_year": EXAM_CONTEXT_2025["year"],
    })


# ============================================================
# 啟動
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
