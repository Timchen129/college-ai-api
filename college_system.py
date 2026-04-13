import os, re, json, time, hashlib, math
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict

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

_rate_store: dict = defaultdict(list)
_chat_rate_store: dict = defaultdict(list)
RATE_LIMIT  = 30    # 每個 IP 每小時最多 30 次分析
RATE_WINDOW = 3600  # 時間窗：1 小時（秒）
CHAT_RATE_LIMIT = 60

def check_chat_rate_limit(ip: str) -> bool:
    now = time.time()
    _chat_rate_store[ip] = [t for t in _chat_rate_store[ip] if now - t < RATE_WINDOW]
    if len(_chat_rate_store[ip]) >= CHAT_RATE_LIMIT:
        return False
    _chat_rate_store[ip].append(now)
    return True

def check_rate_limit(ip: str) -> bool:
    now   = time.time()
    calls = _rate_store[ip]
    # 清除超過時間窗的舊記錄
    _rate_store[ip] = [t for t in calls if now - t < RATE_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        return False
    _rate_store[ip].append(now)
    return True

def get_client_ip() -> str:
    # Render / 一般 reverse proxy 會把真實 IP 放在 X-Forwarded-For
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        # X-Forwarded-For 可能是逗號分隔的多個 IP，取第一個
        return forwarded.split(',')[0].strip()
    return request.remote_addr or '0.0.0.0'

# ============================================================
# 第一階段：科目名稱對照表 & 正規化
# ============================================================

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
# 第二階段：今年考試環境背景
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
# 科系資料庫
# ============================================================

def _normalize_major_types(m: dict) -> dict:
    """
    ✅ 修復：將 majors.json 讀入的數值欄位統一轉為 int，
    防止 JSON 字串型數字（如 "14"）在比較與排序時引發
    'str' > 'int' TypeError。
    """
    for field in ("last_year_cutoff_by_subject", "thresholds"):
        if field in m and isinstance(m[field], dict):
            cleaned = {}
            for k, v in m[field].items():
                if v is None:
                    continue
                try:
                    cleaned[k] = int(v)
                except (ValueError, TypeError):
                    print(f"[WARN] {m.get('school','')} {m.get('major','')} — {field}[{k}] 值 '{v}' 非數字，已略過")
            m[field] = cleaned

    if "quota" in m:
        try:
            m["quota"] = int(m["quota"])
        except (TypeError, ValueError):
            m["quota"] = 50

    if "past_thresholds" in m and isinstance(m["past_thresholds"], dict):
        for yr, thr in m["past_thresholds"].items():
            if isinstance(thr, dict):
                fixed = {}
                for k, v in thr.items():
                    if v is None:
                        continue
                    try:
                        fixed[k] = int(v)
                    except (ValueError, TypeError):
                        pass
                m["past_thresholds"][yr] = fixed

    return m


def load_majors() -> list:
    _dir = os.path.dirname(os.path.abspath(__file__))
    _path = os.path.join(_dir, "majors.json")
    try:
        with open(_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # ✅ 修復：讀入後立即做型別正規化
        data = [_normalize_major_types(m) for m in data]
        print(f"[OK] 讀入 majors.json，共 {len(data)} 筆科系")
        return data
    except FileNotFoundError:
        print("[WARN] majors.json 不存在，使用內建擴充資料集")
    except Exception as e:
        print(f"[WARN] majors.json 讀取失敗：{e}")

    # 內建資料集（值均為 int，不需正規化）
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

# 延遲 5 秒執行，不阻塞啟動
import threading

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

def compute_admission_probability(
    gap: int,
    passed_threshold: bool,
    quota: int = 50,
    applicants: int = 0,
    waitlist_count: int = 0,
    difficulty_trend: str = "stable"
) -> int:
    import math

    if not passed_threshold:
        # 未達門檻：base 25%，每差一分扣 7%
        prob = 25 + gap * 7
        if quota > 100: prob += 3
        elif quota < 30: prob -= 3
        return max(3, min(30, int(prob)))

    # ── 基礎 sigmoid，中心點移至 gap=-1.8（使 gap=0 → 65%）
    k = 0.55
    midpoint = -0.5
    raw = 1 / (1 + math.exp(-k * (gap - midpoint)))
    prob = raw * 100

    # ── 競爭比修正（報考人數 / 錄取名額）
    if applicants > 0 and quota > 0:
        comp = applicants / quota
        if   comp > 12: prob -= 10
        elif comp > 9:  prob -= 7
        elif comp > 6:  prob -= 4
        elif comp > 4:  prob -= 2
        elif comp < 3:  prob += 5
        elif comp < 2:  prob += 8

    # ── 備取深度修正（備取名額 / 錄取名額）
    # 備取越深 → 落榜後候補機會越大 → 實際錄取機率提升
    if waitlist_count > 0 and quota > 0:
        wr = waitlist_count / quota
        if   wr > 0.4: prob += 6
        elif wr > 0.2: prob += 3
        elif wr < 0.05: prob -= 4

    # ── 趨勢修正
    if   difficulty_trend == "rising":  prob -= 5
    elif difficulty_trend == "falling": prob += 5

    # ── 名額規模修正
    if   quota < 30:  prob -= 4
    elif quota > 120: prob += 3

    return max(8, min(95, int(prob)))

def predict_next_year_cutoff(m: dict) -> dict:
    import math

    past = m.get("past_thresholds", {})
    last = m.get("last_year_cutoff_by_subject", {})
    applicants = m.get("applicants", 0)
    quota = m.get("quota", 50)
    trend = m.get("difficulty_trend", "stable")

    # ── 合併所有年份資料
    all_years: dict = {}
    for yr, thr in past.items():
        if isinstance(thr, dict):
            all_years[str(yr)] = {k: int(v) for k, v in thr.items() if v is not None}
    sorted_years = sorted(all_years.keys())
    if sorted_years:
        latest = sorted_years[-1]
        for subj, val in last.items():
            if subj not in all_years.get(latest, {}):
                new_yr = str(int(latest) + 1)
                all_years.setdefault(new_yr, {})[subj] = int(val)
    elif last:
        all_years["114"] = {k: int(v) for k, v in last.items()}

    sorted_years = sorted(all_years.keys())
    if not sorted_years:
        return {}

    all_subjects = set()
    for thr in all_years.values():
        all_subjects.update(thr.keys())

    # 動態從 Gemini 取得，失敗自動 fallback
    difficulty_adj = {
        subj: _DIFFICULTY_PRECOMPUTED.get(subj, get_ai_difficulty_adjustment(subj))
        for subj in all_subjects
    }

    # ── 競爭比對錄取線的影響
    # 競爭比下降（少子化）→ 錄取線微降；上升 → 微升
    comp_adj = 0.0
    if applicants > 0 and quota > 0:
        comp = applicants / quota
        # 與歷史平均競爭比 7x 相比
        comp_adj = (comp - 7) * 0.05  # 每差 1x，影響 0.05 分

    predictions = {}
    for subj in all_subjects:
        subj_data = [
            (yr, all_years[yr][subj])
            for yr in sorted_years
            if subj in all_years[yr]
        ]
        if not subj_data:
            continue
        if len(subj_data) == 1:
            predictions[subj] = subj_data[0][1]
            continue

        # ── 指數加權平均（近年權重更高）
        weights = [math.exp(0.5 * i) for i in range(len(subj_data))]
        total_w = sum(weights)
        weighted_avg = sum(w * v for w, (_, v) in zip(weights, subj_data)) / total_w

        # ── 線性趨勢斜率
        xs = list(range(len(subj_data)))
        ys = [v for _, v in subj_data]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den if den else 0

        # ── difficulty_trend 整體調整
        trend_bonus = {"rising": 0.3, "falling": -0.3, "stable": 0.0}.get(trend, 0.0)

        raw = weighted_avg + slope + difficulty_adj.get(subj, 0.0) + comp_adj + trend_bonus
        predictions[subj] = max(1, min(15, round(raw)))

    return predictions

_difficulty_cache: dict = {}

def get_ai_difficulty_adjustment(subject: str) -> float:
    """
    呼叫 Gemini 取得今年該科目的難度調整係數。
    快取 24 小時避免重複呼叫。
    """
    cache_key = f"difficulty_{subject}_2025"
    if cache_key in _difficulty_cache:
        ts, val = _difficulty_cache[cache_key]
        if time.time() - ts < 86400:  # 24小時有效
            return val

    fallback = {"數學A": -0.5, "英文": -0.3, "國文": 0.0, "自然": -0.2, "社會": 0.0}

    if not GEMINI_AVAILABLE or not genai:
        return fallback.get(subject, 0.0)

    try:
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = (
            f"你是台灣學測分析專家。"
            f"請根據2025年（115學年度）台灣學科能力測驗「{subject}」的實際考試難度，"
            f"相較於2024年（114學年度），給出一個難度調整係數。"
            f"規則：若今年比去年難，係數為負（如-0.5代表預測錄取線下修0.5級）；"
            f"若今年比去年容易，係數為正；持平則為0。"
            f"範圍限制在-2.0到+2.0之間。"
            f"只回傳一個浮點數，不要任何說明或文字。"
        )
        res = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 10, "temperature": 0.1}
        )
        val = float(res.text.strip())
        val = max(-2.0, min(2.0, val))
        _difficulty_cache[cache_key] = (time.time(), val)
        print(f"[AI難度] {subject} 係數：{val}")
        return val
    except Exception as e:
        print(f"[WARN] 難度係數取得失敗 ({subject})：{e}")
        return fallback.get(subject, 0.0)

# ============================================================
# 落點配對
# ============================================================


def generate_ai_comment(m: dict, gap: int, passed_threshold: bool) -> str:
    subj_detail = m.get("subject_detail", {})
    tiebreak    = m.get("tiebreak_subject", "")
    
    # 找出學生在決勝科目的具體分數與錄取線
    student_score = None
    cutoff_score  = None
    if tiebreak and tiebreak in subj_detail:
        student_score = subj_detail[tiebreak].get("student")
        cutoff_score  = subj_detail[tiebreak].get("cutoff")
    
    # 歷年趨勢判斷
    past  = m.get("past_thresholds", {})
    years = sorted(past.keys())
    trend_text = ""
    if len(years) >= 2:
        def avg(thr):
            v = [x for x in thr.values() if isinstance(x, (int, float))]
            return sum(v)/len(v) if v else 0
        delta = avg(past[years[-1]]) - avg(past[years[-2]])
        if delta > 0.3:
            trend_text = f"（門檻連年上升 +{delta:.1f}，今年競爭更激烈）"
        elif delta < -0.3:
            trend_text = f"（門檻近年下滑 {delta:.1f}，對你有利）"
    
    # 組合個人化評語
    if not passed_threshold:
        fails = m.get("failed_thresholds", {})
        fail_details = "、".join(f"{s} 還差 {abs(student_score - req) if student_score else '?'} 分" 
                                  for s, req in fails.items())
        return f"未過最低門檻（{fail_details}），列為衝刺志願。{trend_text}"
    
    if tiebreak and student_score is not None and cutoff_score is not None:
        diff = student_score - cutoff_score
        if diff > 2:
            score_part = f"決勝科目 {tiebreak} 你拿 {student_score}，比去年錄取線 {cutoff_score} 高 {diff} 分，安全邊際足夠"
        elif diff > 0:
            score_part = f"{tiebreak} 你拿 {student_score}，僅比去年錄取線 {cutoff_score} 高 {diff} 分，建議確認今年倍率"
        elif diff == 0:
            score_part = f"{tiebreak} 你拿 {student_score}，恰好與去年錄取線持平，需觀察今年競爭"
        else:
            score_part = f"{tiebreak} 你拿 {student_score}，低於去年錄取線 {cutoff_score} 達 {abs(diff)} 分"
    else:
        score_part = "落點符合歷年錄取區間" if gap >= 0 else f"低於去年錄取線 {abs(gap)} 分"
    
    ai_impact = m.get("ai_impact", "")
    ai_part = ""
    if ai_impact in ("高度受益", "受益"):
        ai_part = "，AI時代優勢明顯"
    elif ai_impact == "部分衝擊":
        ai_part = "，留意AI對部分職位的替代風險"
    
    return f"{score_part}{trend_text}{ai_part}。"



def score_relevance(m_entry: dict, profile: dict) -> float:
    """
    計算科系與使用者偏好的相關性分數（0~1）。
    用於在同一安全層內進行二次排序，讓符合興趣的科系優先。
    """
    score = 0.0
    interests  = profile.get("interests",  "").lower()
    strengths  = profile.get("strengths",  "").lower()
    group      = m_entry.get("group",        "").lower()
    tags       = [t.lower() for t in m_entry.get("industry_tags", [])]
    major_name = m_entry.get("major",       "").lower()
    career     = [c.lower() for c in m_entry.get("career", [])]

    # ── 興趣關鍵字命中 ──
    interest_keywords = {
        "科技工程": ["理工", "工程", "資訊", "電機", "機械", "化工", "材料", "ai", "半導體"],
        "醫療健康": ["醫", "藥", "護理", "公衛", "醫藥", "生醫"],
        "商業金融": ["商", "管理", "財務", "金融", "經濟", "會計","文商"],
        "法律政治": ["法律", "政治", "行政", "公共"],
        "藝術設計": ["藝術", "設計", "建築", "美術", "音樂"],
        "教育社會": ["教育", "師範", "社工", "社會", "心理"],
        "傳播媒體": ["傳播", "新聞", "廣告", "影視"],
    }
    for kw_group, kws in interest_keywords.items():
        if kw_group in interests:
            for kw in kws:
                if kw in group or kw in major_name or any(kw in t for t in tags):
                    score += 0.35
                    break

    # ── industry_tags 直接命中 ──
    for tag in tags:
        if tag in interests or tag in strengths:
            score += 0.15

    # ── career 職涯命中 ──
    for c in career:
        if any(i.strip() in c for i in interests.split("、") if i.strip()):
            score += 0.1

    # ── 就業率加分（高就業率科系有輕微加分）──
    emp = m_entry.get("employment_rate", 0)
    score += min(emp / 1000, 0.1)   # 最多 +0.1

    return min(score, 1.0)

_DIFFICULTY_PRECOMPUTED: dict = {}

def precompute_difficulty():
    """啟動時預計算所有科目難度係數（blocking，確保完成）"""
    global _DIFFICULTY_PRECOMPUTED
    subjects = ["國文", "英文", "數學A", "數學B", "自然", "社會"]
    for subj in subjects:
        _DIFFICULTY_PRECOMPUTED[subj] = get_ai_difficulty_adjustment(subj)
    print(f"[OK] 難度係數預計算完成：{_DIFFICULTY_PRECOMPUTED}")
# ── 伺服器啟動時立即預計算，確保第一次請求不卡頓 ──

try:
    precompute_difficulty()
except Exception as e:
    print(f"[WARN] 難度預計算失敗，將使用 fallback：{e}")

def match_majors(scores: dict, profile: dict = None) -> list:
    """
    落點配對，支援分層配額與使用者偏好排序。

    分層配額（回傳上限）：
      穩上  → 最多 QUOTA_SAFE  筆
      目標  → 最多 QUOTA_TARGET 筆
      挑戰  → 最多 QUOTA_HARD   筆
    同一層內，依「使用者偏好相關性」降冪，再依 gap 降冪。
    """
    if profile is None:
        profile = {}

    MIN_GAP_HARD = -5  # gap < -5 直接排除，太遠沒意義

    all_entries: list = []

    for m in majors_db:
        multipliers: dict = m.get("multipliers", {})
        cutoff_map: dict = {
            k: int(v)
            for k, v in m.get("last_year_cutoff_by_subject", {}).items()
            if v is not None
        }
        thresholds: dict = {
            k: int(v)
            for k, v in m.get("thresholds", {}).items()
            if v is not None
        }
        past: dict = m.get("past_thresholds", {})

        # ── active：學生有分數（>0）且倍率 > 0 的科目 ──
        # 修正：用 scores.get(s, 0) > 0 而非 s in scores，
        # 避免未選考科目（前端未送出）被誤算進決勝科目
        active = {s: w for s, w in multipliers.items() if w > 0 and scores.get(s, 0) > 0}
        if not active:
            continue

        failed_thresholds = {
            subj: req
            for subj, req in thresholds.items()
            if scores.get(subj, 0) < req
        }
        passed_threshold = len(failed_thresholds) == 0

        min_mult = min(active.values())
        tiebreak_subjects = [s for s, w in active.items() if w == min_mult]

        gap = None
        tiebreak_subject = None
        for subj in tiebreak_subjects:
            student_val = int(scores.get(subj, 0))
            cutoff_val  = cutoff_map.get(subj)
            if cutoff_val is None:
                continue
            g = student_val - cutoff_val
            if gap is None or g < gap:
                gap = g
                tiebreak_subject = subj

        # ── Fallback：tiebreak 科目都沒有 cutoff 資料時，
        #    改用 cutoff_map 裡任何一個有資料且學生有填分的科目 ──
        if gap is None:
            for subj, cutoff_val in cutoff_map.items():
                if int(scores.get(subj, 0)) > 0:
                    gap = int(scores.get(subj, 0)) - cutoff_val
                    tiebreak_subject = subj
                    break
        if gap is None:
            continue

        if gap < MIN_GAP_HARD:
            continue

        if not passed_threshold:
            safety = "困難"
        elif gap > 0:
            safety = "穩健"
        elif gap >= -1:
            safety = "挑戰"
        else:
            safety = "困難"

        subject_detail = {}
        for subj, mult in multipliers.items():
            if mult > 0:
                sv = int(scores.get(subj, 0))
                cv = cutoff_map.get(subj)
                subject_detail[subj] = {
                    "student":         sv,
                    "cutoff":          cv,
                    "gap":             (sv - cv) if cv is not None else None,
                    "multiplier":      mult,
                    "is_tiebreak":     (subj == tiebreak_subject),
                    "threshold":       thresholds.get(subj),
                    "below_threshold": sv < thresholds.get(subj, 0) if subj in thresholds else False,
                }

        history_summary = {
            yr: thr 
            for yr, thr in sorted(past.items(), key=lambda x: int(x[0]))
        }
        quota         = m.get("quota", 50)
        admission_prob = compute_admission_probability(
            gap,
            passed_threshold,
            quota=m.get("quota", 50),
            applicants=m.get("applicants", 0),
            waitlist_count=m.get("waitlist_count", 0),
            difficulty_trend=m.get("difficulty_trend", "stable")
        )
        salary_year   = parse_salary_median(m.get("salary_median", 0))
        combined_pr   = calculate_combined_pr({s: scores.get(s, 0) for s in active})
        relevance     = score_relevance(m, profile)

        try:
            ai_comment_val = generate_ai_comment(m, gap, passed_threshold)
        except Exception:
            ai_comment_val = ""

        try:
            predicted_cutoff_val = predict_next_year_cutoff(m)
        except Exception:
            predicted_cutoff_val = {}

        entry = {
            "school":               m["school"],
            "major":                m["major"],
            "group":                m.get("group", ""),
            "tiebreak_subject":     tiebreak_subject,
            "tiebreak_multiplier":  min_mult,
            "gap":                  int(gap),
            "safety":               safety,
            "is_preferred":         passed_threshold,
            "failed_thresholds":    failed_thresholds,
            "subject_detail":       subject_detail,
            "multipliers":          multipliers,
            "past_thresholds":      history_summary,
            "quota":                quota,
            "industry_tags":        m.get("industry_tags", []),
            "admission_prob":       admission_prob,
            "salary_median_raw":    m.get("salary_median", "—"),
            "salary_year_wan":      salary_year,
            "career":               m.get("career", []),
            "ai_impact":            m.get("ai_impact", "未知"),
            "notes":                m.get("notes", ""),
            "overseas_ratio":       m.get("overseas_ratio", 0),
            "license_required":     m.get("license_required", False),
            "double_major_friendly":m.get("double_major_friendly", True),
            "employment_rate":      m.get("employment_rate", 0),
            "combined_pr":          combined_pr,
            "relevance_score":      round(relevance, 3),
            "school_is_national":   "國立" in m["school"],
            "school_region":        school_region(m["school"]),
            # ── 新增欄位 ──
            "status":               safety,
            "threshold":            thresholds,
            "ai_comment":           generate_ai_comment(m, gap, passed_threshold),
            "predicted_cutoff":     predict_next_year_cutoff(m),
            "prediction_available": bool(m.get("past_thresholds")),
        }
        all_entries.append(entry)

    # ── 全量排序：錄取機率 DESC，同機率依 gap DESC（分差接近度）──
    all_entries.sort(key=lambda x: (-x["admission_prob"], -x["gap"]))
    return all_entries


def sort_by_school_pref(matches: list, pref: str) -> list:
    """
    在已分層的 matches 裡，對每一層內部套用學校偏好排序。
    保留穩上 > 目標 > 挑戰的層級順序，只在同層內調整位置。
    """
    if pref == "any":
        return matches

    tier_order = {"穩健": 0, "挑戰": 1, "困難": 2}

    def sort_key(m: dict) -> tuple:
        tier = tier_order.get(m["safety"], 9)
        s = m["school"]
        if pref == "top3":     school_rank = 0 if s in TOP_SCHOOLS else 1
        elif pref == "north":  school_rank = 0 if m["school_region"] == "north" else 1
        elif pref == "south":  school_rank = 0 if m["school_region"] == "south" else 1
        elif pref == "national": school_rank = 0 if m["school_is_national"] else 1
        else:                  school_rank = 0
        return (tier, school_rank, -m["relevance_score"], -int(m["gap"]))

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
# AI Prompt
# ============================================================

def generate_advice(profile: dict, matches: list) -> str:
    """
    純靜態分析：依落點數據 + 時事背景產生 HTML，不呼叫任何外部 API。
    """
    cache_key = make_cache_key(
        "advice_v3",
        scores,  # 直接用外層的 scores 變數
        profile.get("name", ""),
        [(m["school"], m["major"]) for m in matches]
    )
    cached = cache_get(cache_key)
    if cached:
        return cached

    name    = profile.get("name", "同學")
    scores  = profile.get("scores", {})
    abroad  = profile.get("出國意願", "n")

    # ── 1. 志願推薦（前4名）──
    top4 = matches[:4]

    SAFETY_LABEL = {"穩健": "🟢 穩健", "挑戰": "🟡 挑戰", "困難": "🔴 困難"}
    PROB_DESC = {
        97: "接近確定上榜",
        85: "很有把握",
        70: "有競爭力，仍需留意",
        40: "低於去年錄取線，需衡量時勢",
    }

    def prob_desc(p: int) -> str:
        for k in sorted(PROB_DESC, reverse=True):
            if p >= k:
                return PROB_DESC[k]
        return "機會偏低，建議作為備選"

    def trend_note(past: dict, gap: int) -> str:
        """根據歷年趨勢判斷今年預測方向"""
        if not past:
            return ""
        years = sorted(past.keys())
        if len(years) < 2:
            return ""
        last_yr = years[-1]
        prev_yr = years[-2]
        # 比較最近兩年任一科目平均
        def avg_cutoff(thr: dict) -> float:
            vals = [v for v in thr.values() if isinstance(v, (int, float))]
            return sum(vals) / len(vals) if vals else 0
        delta = avg_cutoff(past[last_yr]) - avg_cutoff(past[prev_yr])
        if delta > 0.3:
            return "近年錄取門檻<strong>上升</strong>，競爭加劇，今年門檻預計持平或微升。"
        elif delta < -0.3:
            return "近年錄取門檻<strong>下滑</strong>，今年分數線可能略降，對你有利。"
        else:
            return "近年錄取門檻<strong>穩定</strong>，今年預測與去年相近。"

    recs_html = ""
    for i, m in enumerate(top4, 1):
        label  = SAFETY_LABEL.get(m["safety"], m["safety"])
        prob   = m.get("admission_prob", 0)
        pdesc  = prob_desc(prob)
        tr     = trend_note(m.get("past_thresholds", {}), m["gap"])
        tags   = "、".join(m.get("industry_tags", [])[:3]) or "—"
        salary = m.get("salary_median_raw", "—")
        ai_imp = m.get("ai_impact", "未知")
        gap_str = f"+{m['gap']}" if m["gap"] >= 0 else str(m["gap"])

        # 預測明年落點
        pred = m.get("predicted_cutoff", {})
        pred_str = ""
        if pred:
            pred_parts = [f"{subj} 預測 {val} 級" for subj, val in pred.items()]
            pred_str = f'<br><small style="color:#2980b9">📊 明年落點預測：{"、".join(pred_parts[:3])}</small>'

        fail_str = ""
        if m.get("failed_thresholds"):
            fail_str = "；".join(
                f"{subj} 差 {m['gap'] if m['gap'] < 0 else ''}{abs(scores.get(subj,0) - req)} 分達門檻"
                for subj, req in m["failed_thresholds"].items()
            )
            fail_str = f'<br><span style="color:#c0392b">⚠️ 未達最低門檻：{fail_str}</span>'

        recs_html += f"""
<div style="margin-bottom:12px;padding:10px 14px;border-left:4px solid {'#27ae60' if m['safety']=='穩健' else '#e74c3c'};background:#fafafa;border-radius:4px">
  <strong>{i}. {m['school']} {m['major']}</strong>　{label}　錄取率 <strong>{prob}%</strong>（{pdesc}）<br>
  決勝科目差距：{gap_str} 分　產業：{tags}　年薪中位：{salary}　AI影響：{ai_imp}{fail_str}{pred_str}
  {'<br><small style="color:#555">' + tr + '</small>' if tr else ''}
</div>"""

    # ── 2. 時事環境背景摘要（篩出與學生科系相關的趨勢）──
    interests = profile.get("interests", "").lower()
    all_tags  = set()
    for m in matches[:6]:
        all_tags.update(t.lower() for t in m.get("industry_tags", []))

    relevant_trends = []
    for t in EXAM_CONTEXT_2025["key_trends"]:
        keywords = ["ai", "資工", "電機", "資管", "醫", "半導體", "商管", "考生", "數學", "英文"]
        if any(kw in t.lower() for kw in keywords) or any(tag in t.lower() for tag in all_tags):
            relevant_trends.append(t)
    if not relevant_trends:
        relevant_trends = EXAM_CONTEXT_2025["key_trends"][:3]

    trends_html = "".join(f"<li>{t}</li>" for t in relevant_trends[:4])

    # ── 3. 出國升學 ──
    abroad_html = ""
    if abroad == "y":
        abroad_html = """<h4>🌏 出國升學補充</h4>
<p>若考慮境外研究所，建議優先評估：美國 CS/EE（TOEFL 100+、GRE 320+）、
新加坡 NUS/NTU（英文要求高但獎學金機會多）、日本帝大（學費低但需日文基礎）。
推薦先完成國內申請後再備考 TOEFL。</p>"""

    # ── 4. 產業出路評比 ──
    hot = EXAM_CONTEXT_2025["hot_industries"]
    declining = EXAM_CONTEXT_2025["declining_fields"]
    industry_notes = []
    for m in top4:
        for tag in m.get("industry_tags", []):
            if any(h in tag for h in hot):
                industry_notes.append(f"<strong>{m['major']}</strong> 的 {tag} 領域屬於熱門產業，就業前景佳。")
                break
            if any(d in tag for d in declining):
                industry_notes.append(f"<strong>{m['major']}</strong> 涉及 {tag}，此領域近年承壓，請評估長期發展。")
                break
    industry_html = "".join(f"<li>{n}</li>" for n in industry_notes[:3]) if industry_notes else "<li>建議參考各校系就業統計數據再做決定。</li>"

    # ── 5. 給這位同學的一句話 ──
    safe_cnt      = sum(1 for m in matches if m["safety"] == "穩健")
    challenge_cnt = sum(1 for m in matches if m["safety"] == "挑戰")
    hard_cnt      = sum(1 for m in matches if m["safety"] == "困難")
    if safe_cnt >= 4:
        closing = f"{name}，你的成績在資料庫中有 {safe_cnt} 個穩健志願，基本盤紮實，重點放在挑選最符合興趣的科系，別因保守而可惜了好分數。"
    elif safe_cnt >= 2:
        closing = f"{name}，穩健志願有 {safe_cnt} 個、挑戰 {challenge_cnt} 個，建議 2 個穩健壓底、1-2 個挑戰試試，分散風險。"
    else:
        closing = f"{name}，困難志願比例較高（{hard_cnt} 個），今年考生人數減少對你略為有利，但建議確保至少 1~2 個門檻全達的志願保底。"

    html = f"""
<h3>📊 落點分析報告｜{name}</h3>

<h4>🎯 志願推薦（前 {len(top4)} 名）</h4>
{recs_html}

<h4>📈 115學年度考試環境與趨勢</h4>
<ul>{trends_html}</ul>

<h4>💼 產業前景評估</h4>
<ul>{industry_html}</ul>

{abroad_html}

<h4>💬 給你的一句話</h4>
<p>{closing}</p>
"""

    cache_set(cache_key, html)
    return html

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
    # Rate limit 檢查
    ip = get_client_ip()
    if not check_rate_limit(ip):
        return jsonify({
            "status":  "error",
            "message": f"請求過於頻繁，每小時最多分析 {RATE_LIMIT} 次，請稍後再試。"
        }), 429

    try:
        data    = request.json or {}
        scores  = data.get("scores", {})
        profile = data.get("profile", {})

        scores = normalize_subject_keys(scores)

        # 必填驗證：key 不存在 或 值為 0 均視為缺填
        missing = [s for s in ["國文", "英文"] if scores.get(s, 0) == 0]
        if missing:
            return jsonify({"status": "error", "message": f"缺少科目分數：{', '.join(missing)}"}), 400
        if scores.get("數學A", 0) == 0 and scores.get("數學B", 0) == 0:
            return jsonify({"status": "error", "message": "請至少填入 數學A 或 數學B"}), 400
        # 自然和社會皆為選考，不強制要求
        # 但如果兩者都是 0，給出提示（不擋住）
        if scores.get("自然", 0) == 0 and scores.get("社會", 0) == 0:
            pass  # 允許通過，只是落點會侷限在不需要這兩科的科系

        for subj, val in list(scores.items()):
            try:
                val = int(val)
            except (TypeError, ValueError):
                return jsonify({"status": "error", "message": f"{subj} 分數格式錯誤"}), 400
            if val != 0 and not (1 <= val <= 15):
                return jsonify({"status": "error", "message": f"{subj} 需為 1~15 的學測級分"}), 400
            scores[subj] = val

        # profile["scores"] 在型別轉換後才設定，確保拿到 int
        profile["scores"] = scores

        matches = match_majors(scores, profile)

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
            "safe":      sum(1 for m in matches if m["safety"] == "穩健"),
            "challenge": sum(1 for m in matches if m["safety"] == "挑戰"),
            "hard":      sum(1 for m in matches if m["safety"] == "困難"),
            # 前端中文 key 相容
            "穩健":      sum(1 for m in matches if m["safety"] == "穩健"),
            "挑戰":      sum(1 for m in matches if m["safety"] == "挑戰"),
            "困難":      sum(1 for m in matches if m["safety"] == "困難"),
        }

        ai_result = generate_advice(profile, matches)

        return jsonify({
            "status":  "success",
            "result":  ai_result,
            "matches": matches,
            "summary": summary,
            "exam_context": EXAM_CONTEXT_2025["key_trends"][:3],
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    ip = get_client_ip()
    if not check_chat_rate_limit(ip):
        return jsonify({
            "status": "error",
            "reply":  f"請求過於頻繁，每小時最多對話 {CHAT_RATE_LIMIT} 次，請稍後再試。"
        }), 429

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
        "status":           "ok",
        "model":            MODEL_NAME,
        "gemini_available": GEMINI_AVAILABLE,
        "memory_size":      len(memory_store),
        "cache_size":       len(_cache),
        "majors_count":     len(majors_db),
        "active_sessions":  len(chat_sessions),
        "api_key_set":      bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "exam_year":        EXAM_CONTEXT_2025["year"],
        "rate_tracked_ips": len(_rate_store),      # 加這行
        "chat_tracked_ips": len(_chat_rate_store), # 加這行
    })

@app.route("/ping")
def ping():
    return jsonify({"ok": True}), 200

# ============================================================
# 啟動
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
