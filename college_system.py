import os
import re
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

#寫入所有系所數據
with open('majors.json', 'r', encoding='utf-8') as f:
    majors_db = json.load(f)
# ============================================================
# 初始化
# ============================================================
 
MODEL_NAME = "gemini-1.5-flash"
 
try:
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
except ImportError:
    GEMINI_AVAILABLE = False
 
app = Flask(__name__)
CORS(app)
 
# ============================================================
# 快取層
# ============================================================
 
_cache: dict = {}
CACHE_TTL = 3600  # 1 小時
 
 
def cache_get(key: str):
    item = _cache.get(key)
    if item and time.time() - item["ts"] < CACHE_TTL:
        return item["val"]
    return None
 
 
def cache_set(key: str, val):
    _cache[key] = {"val": val, "ts": time.time()}
 
 
def make_cache_key(*args) -> str:
    raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()
 
 
# ============================================================
# 科系資料庫
# 欄位說明：
#   weights             → 各科倍率（用於加權總分計算，/compare 路由）
#   thresholds          → 各科最低門檻級分（0 = 無門檻）
#   last_year_score     → 去年最低錄取加權總分（供 predict_cutoff 使用）
#   last_year_cutoff_by_subject → 去年倍率篩選時各科最低錄取級分
# ============================================================
 
majors_db = [
    {
        "school": "台灣大學", "major": "電機工程", "group": "理工",
        "weights":   {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "thresholds": {"數學": 12, "自然": 11},
        "last_year_score": 58, "trend": "上升", "stability": "高",
        "last_year_cutoff_by_subject": {"數學": 14, "自然": 13, "英文": 12, "國文": 10},
        "required_abilities": ["數學", "物理"], "bonus_factor": ["英文"],
        "salary_avg": 85000, "career": ["半導體", "IC設計", "AI工程"],
        "employment_rate": 97, "overseas_ratio": 35,
        "industry_outlook": "正面", "ai_impact": "受益",
        "notes": "台積電、聯發科主要來源校系",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "台灣大學", "major": "資訊工程", "group": "理工",
        "weights":   {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "thresholds": {"數學": 13},
        "last_year_score": 60, "trend": "上升", "stability": "高",
        "last_year_cutoff_by_subject": {"數學": 15, "英文": 13, "自然": 12, "國文": 10},
        "required_abilities": ["數學", "邏輯"], "bonus_factor": ["英文"],
        "salary_avg": 90000, "career": ["AI", "軟體工程", "新創"],
        "employment_rate": 98, "overseas_ratio": 40,
        "industry_outlook": "正面", "ai_impact": "受益",
        "notes": "矽谷就業率最高的台灣科系之一",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "清華大學", "major": "資訊工程", "group": "理工",
        "weights":   {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "thresholds": {"數學": 12},
        "last_year_score": 56, "trend": "上升", "stability": "高",
        "last_year_cutoff_by_subject": {"數學": 13, "英文": 12, "自然": 11, "國文": 9},
        "required_abilities": ["數學", "邏輯"], "bonus_factor": ["英文"],
        "salary_avg": 82000, "career": ["AI", "軟體", "半導體"],
        "employment_rate": 96, "overseas_ratio": 30,
        "industry_outlook": "正面", "ai_impact": "受益",
        "notes": "竹科地緣優勢明顯",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "清華大學", "major": "電機工程", "group": "理工",
        "weights":   {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "thresholds": {"數學": 11, "自然": 11},
        "last_year_score": 55, "trend": "持平", "stability": "高",
        "last_year_cutoff_by_subject": {"數學": 13, "自然": 12, "英文": 11, "國文": 9},
        "required_abilities": ["數學", "物理"], "bonus_factor": [],
        "salary_avg": 80000, "career": ["半導體", "IC設計", "電力電子"],
        "employment_rate": 96, "overseas_ratio": 28,
        "industry_outlook": "正面", "ai_impact": "受益",
        "notes": "",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "交通大學", "major": "資訊工程", "group": "理工",
        "weights":   {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "thresholds": {"數學": 11},
        "last_year_score": 54, "trend": "上升", "stability": "高",
        "last_year_cutoff_by_subject": {"數學": 13, "英文": 11, "自然": 10, "國文": 9},
        "required_abilities": ["數學", "邏輯"], "bonus_factor": ["英文"],
        "salary_avg": 80000, "career": ["AI", "網路", "軟體"],
        "employment_rate": 95, "overseas_ratio": 28,
        "industry_outlook": "正面", "ai_impact": "受益",
        "notes": "與清華合校後資源更豐富",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "成功大學", "major": "電機工程", "group": "理工",
        "weights":   {"國文": 1, "英文": 2, "數學": 3, "自然": 3, "社會": 0},
        "thresholds": {"數學": 10, "自然": 10},
        "last_year_score": 52, "trend": "持平", "stability": "高",
        "last_year_cutoff_by_subject": {"數學": 12, "自然": 11, "英文": 10, "國文": 8},
        "required_abilities": ["數學", "物理"], "bonus_factor": [],
        "salary_avg": 75000, "career": ["半導體", "電子業", "IC設計"],
        "employment_rate": 95, "overseas_ratio": 22,
        "industry_outlook": "正面", "ai_impact": "受益",
        "notes": "南科人才搖籃，就業在地化優勢",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "成功大學", "major": "醫學系", "group": "醫",
        "weights":   {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
        "thresholds": {"自然": 13, "英文": 12, "數學": 11},
        "last_year_score": 65, "trend": "持平", "stability": "高",
        "last_year_cutoff_by_subject": {"自然": 14, "英文": 13, "數學": 12, "國文": 11},
        "required_abilities": ["記憶力", "自然"], "bonus_factor": [],
        "salary_avg": 120000, "career": ["醫師", "研究醫學"],
        "employment_rate": 99, "overseas_ratio": 5,
        "industry_outlook": "穩定", "ai_impact": "部分影響",
        "notes": "南台灣最頂尖醫學系，錄取分數極高",
        "license_required": True, "double_major_friendly": False,
    },
    {
        "school": "陽明交大", "major": "醫學系", "group": "醫",
        "weights":   {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
        "thresholds": {"自然": 14, "英文": 13, "數學": 12},
        "last_year_score": 67, "trend": "持平", "stability": "高",
        "last_year_cutoff_by_subject": {"自然": 15, "英文": 14, "數學": 13, "國文": 12},
        "required_abilities": ["記憶力", "自然"], "bonus_factor": [],
        "salary_avg": 125000, "career": ["醫師", "研究醫學"],
        "employment_rate": 99, "overseas_ratio": 8,
        "industry_outlook": "穩定", "ai_impact": "部分影響",
        "notes": "與台大醫學系並列最頂尖",
        "license_required": True, "double_major_friendly": False,
    },
    {
        "school": "台灣大學", "major": "護理系", "group": "醫",
        "weights":   {"國文": 1, "英文": 1, "數學": 1, "自然": 2, "社會": 1},
        "thresholds": {"自然": 8},
        "last_year_score": 48, "trend": "上升", "stability": "高",
        "last_year_cutoff_by_subject": {"自然": 10, "英文": 9, "國文": 8, "數學": 7},
        "required_abilities": ["自然", "耐心"], "bonus_factor": [],
        "salary_avg": 65000, "career": ["護理師", "護理長", "醫療管理"],
        "employment_rate": 98, "overseas_ratio": 10,
        "industry_outlook": "正面", "ai_impact": "低影響",
        "notes": "護理缺工嚴重，薪資近年大幅提升",
        "license_required": True, "double_major_friendly": False,
    },
    {
        "school": "台灣大學", "major": "藥學系", "group": "醫",
        "weights":   {"國文": 1, "英文": 2, "數學": 2, "自然": 3, "社會": 0},
        "thresholds": {"自然": 11, "數學": 10},
        "last_year_score": 58, "trend": "持平", "stability": "高",
        "last_year_cutoff_by_subject": {"自然": 13, "英文": 12, "數學": 11, "國文": 9},
        "required_abilities": ["化學", "自然"], "bonus_factor": [],
        "salary_avg": 75000, "career": ["藥師", "製藥研發", "臨床試驗"],
        "employment_rate": 96, "overseas_ratio": 12,
        "industry_outlook": "穩定", "ai_impact": "部分影響",
        "notes": "藥師執照保障穩定就業",
        "license_required": True, "double_major_friendly": False,
    },
    {
        "school": "台灣大學", "major": "法律系", "group": "社會",
        "weights":   {"國文": 2, "英文": 2, "數學": 1, "自然": 0, "社會": 3},
        "thresholds": {"社會": 11, "國文": 10},
        "last_year_score": 57, "trend": "持平", "stability": "中",
        "last_year_cutoff_by_subject": {"社會": 13, "英文": 12, "國文": 11, "數學": 9},
        "required_abilities": ["閱讀", "邏輯"], "bonus_factor": ["國文"],
        "salary_avg": 70000, "career": ["律師", "法務", "公職"],
        "employment_rate": 88, "overseas_ratio": 15,
        "industry_outlook": "穩定", "ai_impact": "部分影響",
        "notes": "需考律師證照才能發揮最大效益",
        "license_required": True, "double_major_friendly": True,
    },
    {
        "school": "台灣大學", "major": "經濟系", "group": "社會",
        "weights":   {"國文": 1, "英文": 2, "數學": 2, "自然": 0, "社會": 2},
        "thresholds": {"數學": 10},
        "last_year_score": 55, "trend": "上升", "stability": "中",
        "last_year_cutoff_by_subject": {"數學": 12, "英文": 12, "社會": 11, "國文": 9},
        "required_abilities": ["數學", "分析"], "bonus_factor": ["英文"],
        "salary_avg": 72000, "career": ["金融", "顧問", "政策研究"],
        "employment_rate": 90, "overseas_ratio": 25,
        "industry_outlook": "穩定", "ai_impact": "部分影響",
        "notes": "出國深造比例高，適合有研究所規劃者",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "政治大學", "major": "財務管理", "group": "社會",
        "weights":   {"國文": 1, "英文": 2, "數學": 2, "自然": 0, "社會": 2},
        "thresholds": {"數學": 9},
        "last_year_score": 50, "trend": "持平", "stability": "中",
        "last_year_cutoff_by_subject": {"數學": 11, "英文": 11, "社會": 10, "國文": 8},
        "required_abilities": ["數學", "分析"], "bonus_factor": ["英文"],
        "salary_avg": 65000, "career": ["投資銀行", "基金", "企業財務"],
        "employment_rate": 89, "overseas_ratio": 20,
        "industry_outlook": "穩定", "ai_impact": "部分影響",
        "notes": "政大商院品牌效益強，金融業認可度高",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "台灣大學", "major": "外文系", "group": "人文",
        "weights":   {"國文": 2, "英文": 3, "數學": 0, "自然": 0, "社會": 1},
        "thresholds": {"英文": 12},
        "last_year_score": 50, "trend": "下降", "stability": "低",
        "last_year_cutoff_by_subject": {"英文": 13, "國文": 11, "社會": 10},
        "required_abilities": ["英文", "語言"], "bonus_factor": [],
        "salary_avg": 42000, "career": ["翻譯", "外貿", "教師"],
        "employment_rate": 75, "overseas_ratio": 20,
        "industry_outlook": "負面", "ai_impact": "高度衝擊",
        "notes": "AI翻譯工具衝擊傳統翻譯市場，需發展複合能力",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "台灣大學", "major": "數學系", "group": "理工",
        "weights":   {"國文": 1, "英文": 2, "數學": 3, "自然": 2, "社會": 0},
        "thresholds": {"數學": 12},
        "last_year_score": 52, "trend": "上升", "stability": "中",
        "last_year_cutoff_by_subject": {"數學": 13, "英文": 11, "自然": 11, "國文": 9},
        "required_abilities": ["數學", "抽象思維"], "bonus_factor": [],
        "salary_avg": 70000, "career": ["金融工程", "AI研究", "精算"],
        "employment_rate": 88, "overseas_ratio": 35,
        "industry_outlook": "正面", "ai_impact": "受益",
        "notes": "AI時代數學底子是核心競爭力，轉資工/金融門票",
        "license_required": False, "double_major_friendly": True,
    },
    {
        "school": "私立大學", "major": "資訊工程", "group": "理工",
        "weights":   {"國文": 1, "英文": 1, "數學": 2, "自然": 1, "社會": 0},
        "thresholds": {"數學": 7},
        "last_year_score": 40, "trend": "下降", "stability": "低",
        "last_year_cutoff_by_subject": {"數學": 9, "英文": 8, "自然": 7, "國文": 6},
        "required_abilities": ["數學"], "bonus_factor": [],
        "salary_avg": 55000, "career": ["軟體開發", "IT維運"],
        "employment_rate": 82, "overseas_ratio": 5,
        "industry_outlook": "負面", "ai_impact": "高度衝擊",
        "notes": "AI工具普及後，非頂尖資工系基層就業壓力大",
        "license_required": False, "double_major_friendly": False,
    },
    {
        "school": "私立大學", "major": "商業管理", "group": "社會",
        "weights":   {"國文": 2, "英文": 2, "數學": 1, "自然": 0, "社會": 2},
        "thresholds": {},
        "last_year_score": 42, "trend": "下降", "stability": "低",
        "last_year_cutoff_by_subject": {"英文": 8, "社會": 8, "國文": 7, "數學": 6},
        "required_abilities": ["溝通"], "bonus_factor": [],
        "salary_avg": 45000, "career": ["行銷", "業務", "管理"],
        "employment_rate": 80, "overseas_ratio": 3,
        "industry_outlook": "負面", "ai_impact": "高度衝擊",
        "notes": "非頂尖商管出路競爭激烈，入學前需謹慎評估",
        "license_required": False, "double_major_friendly": False,
    },
]
 
# ============================================================
# 向量記憶（RAG）
# ============================================================
 
memory_store: list = []
MAX_MEMORY = 200
 
 
def get_embedding(text: str) -> np.ndarray:
    """取得文字 embedding。若 Gemini 不可用，改用 hash-based 固定向量（可重現）。"""
    if not GEMINI_AVAILABLE:
        # 固定向量：以 MD5 做 seed，確保同一文字每次相同
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.random(768)
    res = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(res["embedding"])
 
 
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0
 
 
def store_memory(text: str, tag: str = "general") -> None:
    vec = get_embedding(text)
    memory_store.append({
        "text": text,
        "vec": vec,
        "tag": tag,
        "time": datetime.now().isoformat()
    })
    if len(memory_store) > MAX_MEMORY:
        memory_store.pop(0)
 
 
def retrieve_memory(query: str, top_k: int = 3, tag: str = None) -> str:
    if not memory_store:
        return ""
    q_vec = get_embedding(query)
    pool = [m for m in memory_store if tag is None or m["tag"] == tag]
    if not pool:
        return ""
    scored = sorted(
        [(cosine_sim(q_vec, m["vec"]), m["text"]) for m in pool],
        reverse=True
    )
    return "\n".join(t for _, t in scored[:top_k])
 
 
# 初始化知識庫（必須在所有函數都定義完之後才呼叫）
store_memory(
    "AI衝擊：基層程式員下降\n半導體：持續擴編\n護理：薪資上升\nAI職缺暴增\n商管出路惡化",
    tag="knowledge"
)
 
# ============================================================
# PR 值計算
# ============================================================
 
# 各科學測級分對應 PR 值（根據歷年五標估算的 mock 分布）
_PR_TABLE = {
    15: 99, 14: 96, 13: 90, 12: 82, 11: 72,
    10: 60,  9: 48,  8: 36,  7: 25,  6: 16,
     5:  9,  4:  5,  3:  2,  2:  1,  1:  0
}
 
 
def calculate_pr(score: int) -> int:
    """根據學測級分（1~15）回傳對應 PR 值（0~99）。"""
    return _PR_TABLE.get(max(1, min(15, score)), 0)
 
 
def calculate_combined_pr(scores: dict) -> int:
    """計算多科組合的平均 PR 值（簡化版）。"""
    if not scores:
        return 0
    return round(sum(calculate_pr(v) for v in scores.values()) / len(scores))
 
 
# ============================================================
# 分數計算 & 落點預測（供 /compare 路由使用）
# ============================================================
 
def calculate_score(scores: dict, weights: dict) -> float:
    """計算加權總分（用於 /compare 路由的舊式比較）。"""
    return sum(scores.get(s, 0) * weights.get(s, 0) for s in weights)
 
 
def predict_cutoff(m: dict) -> float:
    """估算今年最低錄取加權總分。"""
    base = m["last_year_score"]
    base += {"上升": +2, "持平": 0, "下降": -2}.get(m.get("trend", "持平"), 0)
    base += {"高": +1, "中": 0, "低": -1}.get(m.get("stability", "中"), 0)
    base += {"正面": +1, "穩定": 0, "負面": -2}.get(m.get("industry_outlook", "穩定"), 0)
    return round(base, 1)
 
 
# ============================================================
# 落點配對（逐科級分比對版）
# ============================================================
 
def match_majors(scores: dict, min_gap: int = -3) -> list:
    """
    以「倍率篩選決勝科目」邏輯比對學生級分與去年錄取門檻。
    scores: 各科學測級分（1~15）
    min_gap: 最低允許的決勝科目級分差（預設 -3）
    """
    results = []
 
    for m in majors_db:
        weights = m.get("weights", {})
        thresholds = m.get("thresholds", {})
        cutoff_by_subject = m.get("last_year_cutoff_by_subject", {})
 
        # 過濾掉倍率為 0 的科目（不計分科目）
        active_subjects = {s: w for s, w in weights.items() if w > 0}
        if not active_subjects:
            continue
 
        # ── STEP 1：檢定門檻過濾 ──
        if any(scores.get(subj, 0) < req for subj, req in thresholds.items()):
            continue
 
        # ── STEP 2：找出最低倍率（決勝）科目，計算 gap ──
        min_mult = min(active_subjects.values())
        tiebreak_subjects = [s for s, w in active_subjects.items() if w == min_mult]
 
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
 
        # ── STEP 3：安全分類 ──
        if gap >= 2:
            safety = "穩上"
        elif gap >= -1:
            safety = "目標"
        else:
            safety = "衝刺"
 
        # PR 值
        subject_prs = {s: calculate_pr(scores.get(s, 0)) for s in active_subjects}
        combined_pr = calculate_combined_pr({s: scores.get(s, 0) for s in active_subjects})
 
        results.append({
            "school": m["school"],
            "major": m["major"],
            "group": m["group"],
            "tiebreak_subject": tiebreak_subject,
            "tiebreak_multiplier": min_mult,
            "student_tiebreak_score": scores.get(tiebreak_subject, 0),
            "cutoff_tiebreak_score": cutoff_by_subject.get(tiebreak_subject),
            "gap": gap,
            "safety": safety,
            "weights": active_subjects,
            "salary_avg": m.get("salary_avg", 0),
            "employment_rate": m.get("employment_rate", 0),
            "career": m["career"],
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
# Markdown → HTML 轉換
# ============================================================
 
def md_to_html(text: str) -> str:
    """將 Gemini 回傳的 Markdown 轉為安全的 HTML。"""
    # 標題
    text = re.sub(r"^#### (.*)$", r"<h4>\1</h4>", text, flags=re.M)
    text = re.sub(r"^### (.*)$",  r"<h3>\1</h3>",  text, flags=re.M)
    text = re.sub(r"^## (.*)$",   r"<h2>\1</h2>",   text, flags=re.M)
    # 粗體 / 斜體
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         text)
    # 列表：先收集連續 li，再一次包 ul（避免跨 block 的 re.S 問題）
    lines = text.split("\n")
    out, in_list = [], False
    for line in lines:
        m = re.match(r"^[-•] (.*)$", line)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"  <li>{m.group(1)}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(line)
    if in_list:
        out.append("</ul>")
    text = "\n".join(out)
    # 段落
    text = re.sub(r"\n{2,}", "</p><p>", text.strip())
    return f"<p>{text}</p>"
 
 
# ============================================================
# AI 分析建議
# ============================================================
 
def compress_matches(matches: list) -> list:
    return [
        {
            "school": m["school"],
            "major": m["major"],
            "safety": m["safety"],
            "salary_avg": m["salary_avg"],
            "ai_impact": m["ai_impact"],
        }
        for m in matches[:3]
    ]
 
 
def generate_advice(profile: dict, matches: list) -> str:
    cache_key = make_cache_key(profile, [m["major"] for m in matches[:5]])
    cached = cache_get(cache_key)
    if cached:
        return cached
 
    context = (
        retrieve_memory(str(profile), top_k=2)
        + "\n"
        + retrieve_memory("產業趨勢", tag="knowledge")
    )
    abroad = profile.get("出國意願", "n")
    abroad_text = (
        "請提供國外升學策略（美國/英國/日本/新加坡），包含推薦研究所、準備方向、TOEFL/GRE門檻"
        if abroad == "y"
        else "出國升學部分略過"
    )
    top_matches = compress_matches(matches)
 
    prompt = f"""
你是台灣升學顧問（務實、直接）。
 
學生：
{json.dumps(profile, ensure_ascii=False)}
 
推薦科系（前3）：
{json.dumps(top_matches, ensure_ascii=False)}
 
參考背景：
{context}
 
請輸出：
1. 志願排序（標註 穩上/目標/衝刺 + 原因）
2. 能力不匹配警告
3. 英文影響（具體差距）
4. 薪資與出路（短中長期 + AI影響）
5. {abroad_text}
6. 雙主修建議
7. 一句話建議（務實）
 
用條列式，避免廢話
"""
 
    if not GEMINI_AVAILABLE:
        return "<p>⚠️ AI 分析模組未載入，請至 Render Dashboard 設定 GEMINI_API_KEY 環境變數</p>"
 
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        res = model.generate_content(prompt)
        html = md_to_html(res.text)
        cache_set(cache_key, html)
        store_memory(
            f"學生：{profile} 推薦：{[m['major'] for m in top_matches[:3]]}",
            tag="consultation"
        )
        return html
    except Exception as e:
        return f"<p>⚠️ AI 分析失敗：{str(e)}</p>"
 
 
# ============================================================
# 聊天歷史（多輪對話）
# SESSION 自動 TTL 清理，避免記憶體洩漏
# ============================================================
 
SESSION_TTL = 3600  # 1 小時無互動即清除
chat_sessions: dict = {}  # {session_id: {"history": [...], "last_active": float}}
 
 
def get_or_create_session(session_id: str) -> list:
    """取得或建立對話 session，並自動清理過期的 session。"""
    now = time.time()
 
    # 清理過期 session
    expired = [sid for sid, s in chat_sessions.items()
               if now - s["last_active"] > SESSION_TTL]
    for sid in expired:
        del chat_sessions[sid]
 
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": [
                {
                    "role": "user",
                    "parts": [
                        "你是台灣升學諮詢顧問，有 15 年實戰經驗。"
                        "請用繁體中文回答，語氣像朋友而非業務員，回答務實不說廢話，"
                        "每次回答不超過 350 字，善用條列式。"
                        "你熟知：學測落點、各科系出路、薪資行情、AI 時代產業趨勢、"
                        "備審資料撰寫、面試技巧。"
                    ]
                },
                {"role": "model", "parts": ["好的，我準備好了！有任何升學問題都可以問我。"]}
            ],
            "last_active": now,
        }
 
    chat_sessions[session_id]["last_active"] = now
    return chat_sessions[session_id]["history"]
 
 
# ============================================================
# API 路由
# ============================================================
 
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "升志 ScoreWise API v3.1",
        "model": MODEL_NAME,
        "gemini_ready": GEMINI_AVAILABLE,
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
 
        # 驗證分數範圍（學測級分 1~15）
        for subj, val in scores.items():
            if not isinstance(val, (int, float)) or not (1 <= val <= 15):
                return jsonify({
                    "status": "error",
                    "message": f"{subj} 分數需為 1~15 的學測級分，收到：{val}"
                }), 400
 
        matches = match_majors(scores, min_gap=-3)
 
        if not matches:
            return jsonify({
                "status": "success",
                "result": "<p>目前分數較難錄取已知科系，建議補考或調整方向。</p>",
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
        session_id = body.get("session_id", "default")
 
        if not user_message:
            return jsonify({"status": "error", "message": "訊息不能為空"}), 400
        if not GEMINI_AVAILABLE:
            return jsonify({
                "status": "error",
                "message": "AI 模組未啟用，請確認 GEMINI_API_KEY 環境變數"
            }), 503
 
        # 快取 key 包含 session_id，避免跨 session 污染
        cache_key = make_cache_key("chat", session_id, user_message)
        cached = cache_get(cache_key)
        if cached:
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
 
        # 保留最近 20 輪對話（系統 prompt 2 則 + 最新 20 則）
        if len(history) > 22:
            history[2:4] = []
 
        store_memory(f"Q:{user_message} A:{reply[:120]}", tag="chat")
        cache_set(cache_key, reply)
 
        return jsonify({"status": "ok", "reply": reply})
 
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
 
@app.route("/majors", methods=["GET"])
def get_majors():
    group = request.args.get("group")
    school = request.args.get("school")
    data = majors_db
    if group:
        data = [m for m in data if m.get("group") == group]
    if school:
        data = [m for m in data if m.get("school") == school]
    # 不回傳 vec（若未來 majors_db 有快取 vec）
    safe = [{k: v for k, v in m.items() if k != "vec"} for m in data]
    return jsonify({"status": "ok", "data": safe, "count": len(safe)})
 
 
@app.route("/predict/<school>/<major>", methods=["GET"])
def predict_single(school, major):
    target = next(
        (m for m in majors_db if m["school"] == school and m["major"] == major),
        None
    )
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
    try:
        body = request.json or {}
        scores = body.get("scores", {})
        majors = body.get("majors", [])
 
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
                **{k: v for k, v in target.items() if k not in ("weights", "vec")},
                "predicted_cutoff": cutoff,
                "student_score": round(student_score, 1) if student_score is not None else None,
                "gap": round(student_score - cutoff, 1) if student_score is not None else None,
            })
 
        return jsonify({"status": "ok", "comparison": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
 
 
@app.route("/salary-insight", methods=["GET"])
def salary_insight():
    group = request.args.get("group")
    data = majors_db if not group else [m for m in majors_db if m.get("group") == group]
    ranked = sorted(data, key=lambda m: m.get("salary_avg", 0), reverse=True)
    result = [
        {
            "school": m["school"],
            "major": m["major"],
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)