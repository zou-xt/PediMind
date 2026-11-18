import streamlit as st
import streamlit.components.v1 as components
import os, time, json, base64, html, sqlite3, uuid
from datetime import datetime
from contextlib import contextmanager
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI

# ================= 1. 环境与 API =================
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

try:
    API_KEY = st.secrets["ZHIPU_API_KEY"]
except Exception:
    API_KEY = "sk-你的真实智谱AI密钥" # 请替换为真实的 API Key

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 2. 四大病例 =================
CASES_CONFIG = {
    "laryngitis": {
        "name": "2岁患儿的犬吠声", "disease": "急性感染性喉炎", "patient": "小雨，2岁3个月，女", "difficulty": "⭐⭐⭐⭐",
        "description": "深夜急诊，患儿突发呼吸困难，家长极度焦虑。",
        "audio": "stridor.mp3", "images": {"犬吠样咳嗽": "dog_cough.jpg", "吸气性喉鸣": "stridor.jpg", "三凹征": "three_depressions.jpg"},
        "clues": ["犬吠样咳嗽", "夜间加重", "吸气性喉鸣", "白天感冒史", "三凹征"],
        "ai_role": "2岁急性喉炎患儿的妈妈。极度焦虑、自责。",
        "ai_rules": "1.绝对禁止说出任何医学术语！只能说大白话(如'咳起来像小狗叫'、'胸口凹进去一块')。2.绝对禁止给出任何诊断结论！3.每次回复控制在50字以内。",
        "diagnosis": {"A": "急性感染性喉炎", "B": "急性会厌炎", "C": "气道异物", "D": "支气管哮喘"},
        "crisis": {
            "A": {"label": "保持坐位，面罩吸氧，布地奈德+肾上腺素雾化", "type": "correct", "disease": -30, "reply": "（雾化后呼吸平稳，面色转红润）"},
            "B": {"label": "平卧位，急送放射科拍颈部侧位片", "type": "invalid", "disease": 20, "reply": "（平卧后哭闹加剧，喉鸣音加重，出现呼吸暂停）"},
            "C": {"label": "静脉推注大剂量地塞米松 + 广谱抗生素", "type": "overuse", "disease": 10, "reply": "（药物过量导致患儿心率增快，烦躁）"},
            "D": {"label": "强行按压患儿，用压舌板检查咽喉部", "type": "harmful", "disease": 40, "reply": "（刺激喉部，突发喉痉挛，面色青紫）"}
        }
    },
    "foreign_body": {
        "name": "1岁患儿的剧烈呛咳", "disease": "气道异物", "patient": "童童，1岁8个月，男", "difficulty": "⭐⭐⭐⭐⭐",
        "description": "进食花生后突发剧烈呛咳，气道部分梗阻，生死一线。",
        "audio": "wheezing.mp3", "images": {"剧烈呛咳": "choking.jpg"},
        "clues": ["进食呛咳史", "剧烈呛咳", "呼吸音不对称", "面色发绀"],
        "ai_role": "1岁气道异物患儿的妈妈。极度恐慌、语无伦次。",
        "ai_rules": "1.只能说大白话(如'吃花生突然咳得整个人弓起来'、'脸憋紫了')。2.禁止给出诊断结论。3.回复50字以内。",
        "diagnosis": {"A": "急性感染性喉炎", "B": "急性会厌炎", "C": "气道异物", "D": "支气管哮喘"},
        "crisis": {
            "A": {"label": "立即施行海姆立克急救法/拍背压胸", "type": "correct", "disease": -30, "reply": "（异物排出，呼吸恢复）"},
            "B": {"label": "用手指盲目抠挖咽喉", "type": "harmful", "disease": 40, "reply": "（异物被推得更深，呼吸完全停止）"},
            "C": {"label": "平卧位，吸氧，等待X线结果", "type": "invalid", "disease": 20, "reply": "（等待过程中，患儿窒息加重）"},
            "D": {"label": "使用镇静剂让患儿安静", "type": "harmful", "disease": 35, "reply": "（呼吸抑制，血氧骤降）"}
        }
    },
    "asthma": {
        "name": "4岁患儿的反复喘息", "disease": "支气管哮喘急性发作", "patient": "安安，4岁，女", "difficulty": "⭐⭐⭐",
        "description": "既往哮喘病史，本次急性发作，需紧急平喘。",
        "audio": "wheezing.mp3", "images": {"喘息": "wheezing.jpg"},
        "clues": ["既往哮喘史", "过敏史", "呼气性呼吸困难", "双肺哮鸣音"],
        "ai_role": "4岁哮喘患儿的妈妈。疲惫、担忧。",
        "ai_rules": "1.说大白话(如'她呼气很费力'、'以前也喘过')。2.禁止给出诊断结论。3.回复50字以内。",
        "diagnosis": {"A": "急性感染性喉炎", "B": "急性会厌炎", "C": "气道异物", "D": "支气管哮喘"},
        "crisis": {
            "A": {"label": "吸氧，雾化吸入支气管舒张剂(沙丁胺醇)", "type": "correct", "disease": -30, "reply": "（喘息缓解，肺部哮鸣音减少）"},
            "B": {"label": "静脉滴注抗生素", "type": "overuse", "disease": 10, "reply": "（抗生素对哮喘无效，延误了平喘时机）"},
            "C": {"label": "平卧位，高流量吸氧", "type": "invalid", "disease": 15, "reply": "（平卧加重呼吸困难，患儿端坐呼吸）"},
            "D": {"label": "使用镇静剂", "type": "harmful", "disease": 40, "reply": "（呼吸抑制，病情急剧恶化）"}
        }
    },
    "epiglottitis": {
        "name": "3岁患儿的高热与流涎", "disease": "急性会厌炎", "patient": "乐乐，3岁，男", "difficulty": "⭐⭐⭐⭐⭐",
        "description": "高危气道急症，患儿拒绝平卧，严禁强行查体。",
        "audio": "stridor.mp3", "images": {"三凹征": "three_depressions.jpg"},
        "clues": ["高热", "流涎", "吞咽困难", "拒绝平卧", "无犬吠样咳嗽"],
        "ai_role": "3岁会厌炎患儿的妈妈。极度恐惧。",
        "ai_rules": "1.说大白话(如'一直流口水'、'一躺下就喘不上气')。2.禁止给出诊断结论。3.回复50字以内。",
        "diagnosis": {"A": "急性感染性喉炎", "B": "急性会厌炎", "C": "气道异物", "D": "支气管哮喘"},
        "crisis": {
            "A": {"label": "保持坐位，吸氧，静脉使用糖皮质激素，准备气管切开/插管", "type": "correct", "disease": -30, "reply": "（气道水肿减轻，准备进一步气道管理）"},
            "B": {"label": "强行使用压舌板检查咽喉部", "type": "harmful", "disease": 50, "reply": "（诱发喉痉挛，呼吸骤停！）"},
            "C": {"label": "平卧位，给予镇静剂", "type": "harmful", "disease": 40, "reply": "（平卧+镇静剂，迅速导致窒息）"},
            "D": {"label": "等待放射科拍颈部侧位片", "type": "invalid", "disease": 20, "reply": "（等待过程中气道完全梗阻）"}
        }
    }
}

# ================= 3. 通用成就 =================
GLOBAL_ACHIEVEMENTS = {
    "first_inquiry": {"icon": "🩺", "name": "初次问诊", "desc": "迈出问诊第一步",
                      "check": lambda ss, c: any(m.get("role") == "user" for m in getattr(ss, "messages", []))},
    "perfect_score": {"icon": "🏆", "name": "急诊之光", "desc": "单局得分 ≥ 80",
                      "check": lambda ss, c: getattr(ss, "final_score", 0) >= 80},
    "all_rounder":   {"icon": "📚", "name": "全能住院医", "desc": "通关全部 4 个病例",
                      "check": lambda ss, c: len(getattr(ss, "completed_cases", set())) >= 4},
    "empathy":       {"icon": "💬", "name": "医者仁心", "desc": "用共情方式交代病情",
                      "check": lambda ss, c: getattr(ss, "score_empathy", 0) > 0},
}

# ================= 4. 病例专属成就 =================
CASE_ACHIEVEMENTS = {
    "laryngitis": {
        "bark_cough":   {"icon": "🐕", "name": "犬吠辨识者", "desc": "识别出「犬吠样咳嗽」",
                         "check": lambda ss, c: "犬吠样咳嗽" in getattr(ss, "unlocked_clues", [])},
        "calm_parent":  {"icon": "🤝", "name": "安抚专家",   "desc": "信任值提升至 ≥ 50",
                         "check": lambda ss, c: getattr(ss, "trust_score", 0) >= 50},
        "sit_and_neb":  {"icon": "🪑", "name": "坐位雾化师", "desc": "成功抢救，患儿转危为安",
                         "check": lambda ss, c: getattr(ss, "game_over", False)
                                                and getattr(ss, "crisis_correct_count", 0) >= 1
                                                and getattr(ss, "disease_progress", 100) < 100},
        "night_warrior":{"icon": "🌙", "name": "深夜守护者", "desc": "以病情度 < 50 通关",
                         "check": lambda ss, c: getattr(ss, "game_over", False)
                                                and getattr(ss, "disease_progress", 100) < 50},
    },
    "foreign_body": {
        "peanut_detective": {"icon": "🥜", "name": "花生侦探",   "desc": "识别出「进食呛咳史」",
                             "check": lambda ss, c: "进食呛咳史" in getattr(ss, "unlocked_clues", [])},
        "heimlich_master":  {"icon": "🤲", "name": "海姆立克大师","desc": "急救处置正确 ≥ 2 次",
                             "check": lambda ss, c: getattr(ss, "crisis_correct_count", 0) >= 2},
        "rapid_response":   {"icon": "⚡", "name": "极速救援",   "desc": "以病情度 ≤ 70 通关",
                             "check": lambda ss, c: getattr(ss, "game_over", False)
                                                    and getattr(ss, "disease_progress", 100) <= 70},
        "clear_airway":     {"icon": "💨", "name": "气道清道夫", "desc": "以病情度 < 60 通关",
                             "check": lambda ss, c: getattr(ss, "game_over", False)
                                                    and getattr(ss, "disease_progress", 100) < 60},
    },
    "asthma": {
        "history_hunter": {"icon": "📋", "name": "病史侦探", "desc": "识别出「既往哮喘史」",
                           "check": lambda ss, c: "既往哮喘史" in getattr(ss, "unlocked_clues", [])},
        "nebulizer_pro":  {"icon": "🌬️", "name": "雾化专家", "desc": "急救处置正确 ≥ 2 次",
                           "check": lambda ss, c: getattr(ss, "crisis_correct_count", 0) >= 2},
        "steady_breath":  {"icon": "🫁", "name": "呼吸守护", "desc": "以病情度 < 50 通关",
                           "check": lambda ss, c: getattr(ss, "game_over", False)
                                                  and getattr(ss, "disease_progress", 100) < 50},
        "rational_drug":  {"icon": "💊", "name": "合理用药", "desc": "全程未出现抗生素滥用",
                           "check": lambda ss, c: getattr(ss, "game_over", False)
                                                  and all("抗生素" not in str(x) for x in getattr(ss, "penalty_log", []))},
    },
    "epiglottitis": {
        "drool_watcher": {"icon": "💧", "name": "流涎观察员", "desc": "识别出「流涎」",
                          "check": lambda ss, c: "流涎" in getattr(ss, "unlocked_clues", [])},
        "no_flat":       {"icon": "🚫", "name": "拒绝平卧",   "desc": "识别出「拒绝平卧」",
                          "check": lambda ss, c: "拒绝平卧" in getattr(ss, "unlocked_clues", [])},
        "safe_exam":     {"icon": "🛡️", "name": "安全查体",   "desc": "全程未使用压舌板刺激咽喉",
                          "check": lambda ss, c: getattr(ss, "game_over", False)
                                                 and all("压舌板" not in str(x) for x in getattr(ss, "penalty_log", []))},
        "high_risk_hero":{"icon": "👑", "name": "高危挑战者", "desc": "以困难模式通关本病例",
                          "check": lambda ss, c: getattr(ss, "game_over", False)
                                                 and getattr(ss, "game_mode", "") == "hard"},
    },
}

def flatten_achievements():
    flat = {f"global:{k}": v for k, v in GLOBAL_ACHIEVEMENTS.items()}
    for cid, achs in CASE_ACHIEVEMENTS.items():
        for k, v in achs.items():
            flat[f"{cid}:{k}"] = v
    return flat

# ================= 5. SQLite 持久化 =================
DB_PATH = "pedimind_records.db"

@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with db() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT, player_name TEXT,
            case_id TEXT, case_name TEXT, disease TEXT, mode TEXT,
            final_score INTEGER,
            score_inquiry INTEGER, score_diagnosis INTEGER,
            score_emergency INTEGER, score_empathy INTEGER,
            disease_progress INTEGER, survived INTEGER,
            mistakes TEXT, achievements TEXT,
            created_at TEXT
        )""")
init_db()

def save_record(rec: dict):
    with db() as c:
        c.execute("""INSERT INTO records
            (player_id,player_name,case_id,case_name,disease,mode,
             final_score,score_inquiry,score_diagnosis,score_emergency,score_empathy,
             disease_progress,survived,mistakes,achievements,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec["player_id"], rec["player_name"], rec["case_id"], rec["case_name"],
             rec["disease"], rec["mode"], rec["final_score"], rec["score_inquiry"],
             rec["score_diagnosis"], rec["score_emergency"], rec["score_empathy"],
             rec["disease_progress"], rec["survived"], rec["mistakes"],
             rec["achievements"], rec["created_at"]))

def fetch_my_records(player_id: str):
    with db() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM records WHERE player_id=? ORDER BY id DESC",
                         (player_id,)).fetchall()
    return [dict(r) for r in rows]

def fetch_all_records():
    with db() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

# ================= 6. 页面与状态 =================
st.set_page_config(page_title="PediMind · 急诊风云", page_icon="🏥", layout="wide")

# 默认改为白天模式
if "theme" not in st.session_state: st.session_state.theme = "day"
if "screen" not in st.session_state: st.session_state.screen = "start"
if "current_case" not in st.session_state: st.session_state.current_case = None
if "game_mode" not in st.session_state: st.session_state.game_mode = None
if "player_name" not in st.session_state: st.session_state.player_name = "实习医生"
if "achievements" not in st.session_state: st.session_state.achievements = []
if "completed_cases" not in st.session_state: st.session_state.completed_cases = set()

if "player_id" not in st.session_state:
    pid = st.query_params.get("pid")
    if pid and pid.strip():
        st.session_state.player_id = pid.strip()
    else:
        st.session_state.player_id = str(uuid.uuid4())[:8]
        st.query_params["pid"] = st.session_state.player_id

# 全局字体定义
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Sans+SC:wght@400;500;700;900&display=swap');"

if st.session_state.theme == "night":
    theme_css = f"""<style>
        {FONT_IMPORT}
        .stApp {{ background: radial-gradient(circle at 50% 0%, #1b263b 0%, #0d1b2a 70%) !important; color: #e2e8f0 !important; font-family: 'Noto Sans SC', sans-serif !important; }}
        p, span, div, label, h1, h2, h3, h4, h5, h6 {{ color: #e2e8f0 !important; font-family: 'Noto Sans SC', sans-serif !important; }}
        section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%) !important; }}
        div[data-testid="stChatInput"] {{ background-color: #ffffff !important; border-radius: 12px !important; }}
        div[data-testid="stChatInput"] textarea {{ color: #4a5568 !important; font-family: 'Noto Sans SC', sans-serif !important; }}
        .stButton>button {{ border-radius: 12px !important; border: 1px solid rgba(0, 180, 216, 0.4) !important; background: rgba(0, 180, 216, 0.1) !important; color: #e2e8f0 !important; font-family: 'Noto Sans SC', sans-serif !important; }}
        .stButton>button:hover {{ background: rgba(0, 180, 216, 0.3) !important; }}
        .status-badge {{ background: rgba(255, 255, 255, 0.05); border-radius: 14px; padding: 15px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); }}
        .case-card {{ padding: 25px; border-radius: 16px; margin-bottom: 20px; background: rgba(0, 180, 216, 0.05); border: 1px solid rgba(0, 180, 216, 0.3); }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: rgba(0, 180, 216, 0.3) !important; border-radius: 16px !important; padding: 15px !important; background: rgba(0,0,0,0.2); }}
    </style>"""
else:
    theme_css = f"""<style>
        {FONT_IMPORT}
        .stApp {{ background-color: #f0f4f8 !important; color: #1a202c !important; font-family: 'Inter', 'Noto Sans SC', sans-serif !important; }}
        p, span, div, label, h1, h2, h3, h4, h5, h6 {{ color: #1a202c !important; font-family: 'Inter', 'Noto Sans SC', sans-serif !important; }}
        section[data-testid="stSidebar"] {{ background-color: #ffffff !important; border-right: 1px solid #e2e8f0; }}
        .stButton>button {{ border-radius: 12px !important; border: 1px solid #0077b6 !important; background: #ffffff !important; color: #0077b6 !important; font-family: 'Inter', 'Noto Sans SC', sans-serif !important; font-weight: 600 !important; }}
        .stButton>button:hover {{ background: #e0f2fe !important; }}
        .status-badge {{ background: #ffffff; border-radius: 14px; padding: 15px; text-align: center; border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
        .case-card {{ padding: 25px; border-radius: 16px; margin-bottom: 20px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: #e2e8f0 !important; border-radius: 16px !important; padding: 15px !important; background: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
    </style>"""
st.markdown(theme_css, unsafe_allow_html=True)

# ================= 侧边栏导航 =================
with st.sidebar:
    st.markdown("### 🎮 系统菜单")
    if st.session_state.screen != "start":
        if st.button("🏠 返回主菜单", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "player_id", "player_name", "achievements", "completed_cases"]:
                    del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()

    st.markdown("---")
    st.markdown("### 📊 数据与成就")
    if st.button("📕 我的错题本", use_container_width=True):
        st.session_state.screen = "profile"; st.rerun()
    if st.button("📊 全服统计", use_container_width=True):
        st.session_state.screen = "stats"; st.rerun()
    if st.button("🏅 成就殿堂", use_container_width=True):
        st.session_state.screen = "achievements"; st.rerun()

    st.markdown("---")
    st.markdown("### 🎨 主题设置")
    if st.button("☀️ 白天模式" if st.session_state.theme == "night" else "🌙 夜晚模式", use_container_width=True):
        st.session_state.theme = "day" if st.session_state.theme == "night" else "night"
        st.rerun()

    with st.expander("🔑 换设备？恢复存档"):
        recover = st.text_input("专属 ID", max_chars=8, key="recover_id")
        if st.button("恢复", key="recover_btn") and recover.strip():
            new_id = recover.strip()
            st.session_state.player_id = new_id
            st.query_params["pid"] = new_id
            st.success(f"✅ 已切换到 ID：`{new_id}`"); st.rerun()

# ================= 7. 辅助函数 =================
@st.cache_data(show_spinner=False)
def load_avatar(path: str) -> str:
    if not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{b64}"

def render_message(role: str, content: str):
    safe = html.escape(str(content)).replace("\n", "<br>")
    night = st.session_state.theme == "night"
    AVATAR_SIZE = "38px"

    def avatar_html(img_path, fallback_emoji, fallback_bg):
        src = load_avatar(img_path)
        if src:
            return (f'<img src="{src}" style="width:{AVATAR_SIZE}; height:{AVATAR_SIZE}; '
                    f'border-radius:50%; object-fit:cover; flex-shrink:0; '
                    f'box-shadow:0 1px 3px rgba(0,0,0,0.2);">')
        return (f'<div style="width:{AVATAR_SIZE}; height:{AVATAR_SIZE}; border-radius:50%; '
                f'background:{fallback_bg}; color:#fff; display:flex; align-items:center; '
                f'justify-content:center; flex-shrink:0; font-size:18px;">{fallback_emoji}</div>')

    if role == "user":
        bg = "#95ec69" if not night else "#3eb575"
        fg = "#111111" if not night else "#ffffff"
        av = avatar_html("static/doctor.png", "👨‍⚕️", "#4a90d9")
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-end; align-items:flex-start; margin:8px 0;">
            <div style="background:{bg}; color:{fg}; padding:10px 14px; border-radius:14px;
                        border-top-right-radius:3px; max-width:75%; word-break:break-word;
                        font-size:15px; line-height:1.6; box-shadow:0 1px 2px rgba(0,0,0,0.12);">
                {safe}
            </div>
            <div style="margin-left:8px;">{av}</div>
        </div>""", unsafe_allow_html=True)
    else:
        bg = "#ffffff" if not night else "rgba(255,255,255,0.12)"
        fg = "#111111" if not night else "#e2e8f0"
        av = avatar_html("static/mom.png", "👩", "#f6a623")
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-start; align-items:flex-start; margin:8px 0;">
            <div style="margin-right:8px;">{av}</div>
            <div style="background:{bg}; color:{fg}; padding:10px 14px; border-radius:14px;
                        border-top-left-radius:3px; max-width:75%; word-break:break-word;
                        font-size:15px; line-height:1.6; box-shadow:0 1px 2px rgba(0,0,0,0.12);">
                {safe}
            </div>
        </div>""", unsafe_allow_html=True)

def render_monitor(disease_progress: int):
    # 更逼真的监护仪排版
    if disease_progress < 40:
        hr, spo2, rr, status = 120, 97, 28, "normal"
    elif disease_progress < 70:
        hr, spo2, rr, status = 140, 93, 35, "warn"
    elif disease_progress < 90:
        hr, spo2, rr, status = 160, 88, 45, "danger"
    else:
        hr, spo2, rr, status = 180, 82, 55, "critical"

    html_code = f"""
    <!DOCTYPE html><html><head><style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      body {{ background:#050a0f; overflow:hidden; font-family:'Courier New',monospace; }}
      .monitor {{ position:relative; width:100%; height:210px; background:#050a0f;
                 border-radius:8px; border:2px solid #1a1a1a; overflow:hidden;
                 box-shadow: inset 0 0 30px rgba(0,0,0,0.8); display:flex; }}
      .wave-area {{ flex: 1; position: relative; }}
      canvas {{ display:block; width:100%; height:100%; }}
      .vitals {{ width: 100px; background: #050a0f; padding: 10px; display: flex; flex-direction: column; justify-content: center; border-left: 1px solid #1a1a1a; }}
      .vital-item {{ margin-bottom: 12px; line-height: 1; }}
      .vital-label {{ font-size: 12px; color: #888; letter-spacing: 1px; font-weight: bold; margin-bottom: 2px; }}
      .vital-value {{ font-size: 32px; font-weight: bold; font-family: 'Courier New', monospace; }}
      .vital-unit  {{ font-size: 11px; color: #888; margin-left: 4px; }}
      
      .c-green {{ color:#39ff14; text-shadow: 0 0 10px rgba(57,255,20,0.5); }} 
      .c-yellow {{ color:#ffcc00; text-shadow: 0 0 10px rgba(255,204,0,0.5); }} 
      .c-red {{ color:#ff3b3b; text-shadow: 0 0 10px rgba(255,59,59,0.5); }}
      .blink {{ animation:blink 0.8s infinite; }}
      @keyframes blink {{ 50% {{ opacity:0.3; }} }}
    </style></head><body>
      <div class="monitor">
        <div class="wave-area"><canvas id="ecg"></canvas></div>
        <div class="vitals">
          <div class="vital-item"><div class="vital-label">HR</div><div id="hr" class="vital-value">--</div><div class="vital-unit">bpm</div></div>
          <div class="vital-item"><div class="vital-label">SpO₂</div><div id="spo2" class="vital-value">--</div><div class="vital-unit">%</div></div>
          <div class="vital-item"><div class="vital-label">RR</div><div id="rr" class="vital-value">--</div><div class="vital-unit">/min</div></div>
        </div>
      </div>
    <script>
      const HR={hr}, SPO2={spo2}, RR={rr}, STATUS="{status}";
      const color = (STATUS==="critical"||STATUS==="danger") ? "#ff3b3b"
                  : STATUS==="warn" ? "#ffcc00" : "#39ff14";
      const canvas = document.getElementById('ecg');
      const ctx = canvas.getContext('2d');
      let W, H;
      function resize() {{
        const r = canvas.parentElement.getBoundingClientRect();
        canvas.width = r.width * devicePixelRatio;
        canvas.height = r.height * devicePixelRatio;
        ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
        W = r.width; H = r.height;
      }}
      window.addEventListener('resize', resize); resize();

      // 真实的 P-QRS-T 波形合成
      function ecgSample(p) {{
        let v = 0;
        if (p > 0.1 && p < 0.18) v += 0.12 * Math.sin((p - 0.1) / 0.08 * Math.PI); // P波
        if (p > 0.22 && p < 0.24) v -= 0.15; // Q波
        if (p > 0.24 && p < 0.26) v += 1.0;  // R波
        if (p > 0.26 && p < 0.28) v -= 0.25; // S波
        if (p > 0.35 && p < 0.55) v += 0.25 * Math.sin((p - 0.35) / 0.2 * Math.PI); // T波
        
        // 基线漂移
        v += 0.02 * Math.sin(p * Math.PI * 2);
        return v;
      }}
      
      const BUFFER = 600;
      const data = new Array(BUFFER).fill(0);
      let phase = 0;
      const cycleDur = 60 / HR;
      const sampleInterval = 0.008;
      const perFrame = 2;
      const dPhase = (sampleInterval / cycleDur) * perFrame;
      
      function drawGrid() {{
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = "rgba(57,255,20,0.08)";
        for (let x=0; x<W; x+=10) {{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }}
        for (let y=0; y<H; y+=10) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }}
        ctx.strokeStyle = "rgba(57,255,20,0.2)";
        for (let x=0; x<W; x+=50) {{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }}
        for (let y=0; y<H; y+=50) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }}
      }}
      
      function drawWave() {{
        const step = W / BUFFER, midY = H * 0.65, amp = H * 0.55;
        ctx.beginPath();
        ctx.strokeStyle = color; ctx.lineWidth = 2.5;
        ctx.shadowBlur = 8; ctx.shadowColor = color;
        for (let i=0; i<BUFFER; i++) {{
          const x = i*step, y = midY - data[i]*amp;
          i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }}
        ctx.stroke(); ctx.shadowBlur = 0;
      }}
      
      function loop() {{
        for (let k=0; k<perFrame; k++) {{
          phase = (phase + dPhase) % 1;
          let v = ecgSample(phase);
          if (STATUS==="danger"||STATUS==="critical") v += (Math.random()-0.5)*0.1;
          data.push(v); data.shift();
        }}
        ctx.clearRect(0,0,W,H);
        drawGrid(); drawWave();
        requestAnimationFrame(loop);
      }}
      requestAnimationFrame(loop);
      
      function jit(base, amt) {{ return Math.round(base + (Math.random()-0.5)*amt); }}
      function updateVitals() {{
        const cls = (STATUS==="critical"||STATUS==="danger") ? "vital-value c-red blink"
                  : STATUS==="warn" ? "vital-value c-yellow" : "vital-value c-green";
        const hrEl=document.getElementById('hr'), spEl=document.getElementById('spo2'), rrEl=document.getElementById('rr');
        hrEl.textContent = jit(HR, 2); spEl.textContent = jit(SPO2, 1); rrEl.textContent = jit(RR, 1);
        hrEl.className = cls; spEl.className = cls; rrEl.className = cls;
      }}
      updateVitals(); setInterval(updateVitals, 1000);
    </script></body></html>
    """
    components.html(html_code, height=220, scrolling=False)

def _try_unlock(key: str):
    if key not in st.session_state.achievements:
        st.session_state.achievements.append(key)

def check_achievements(case, case_id: str):
    ss = st.session_state
    if getattr(ss, "game_over", False) and getattr(ss, "disease_progress", 100) < 100:
        ss.completed_cases.add(case_id)
    for aid, a in GLOBAL_ACHIEVEMENTS.items():
        try:
            if a["check"](ss, case): _try_unlock(f"global:{aid}")
        except Exception: pass
    for aid, a in CASE_ACHIEVEMENTS.get(case_id, {}).items():
        try:
            if a["check"](ss, case): _try_unlock(f"{case_id}:{aid}")
        except Exception: pass

def achievement_total() -> int:
    return len(flatten_achievements())

def achievement_unlocked_count() -> int:
    return len(st.session_state.get("achievements", []))

def render_achievement_grid(ach_dict: dict, prefix: str, cols_per_row: int = 2):
    unlocked = st.session_state.get("achievements", [])
    items = list(ach_dict.items())
    cols = st.columns(cols_per_row)
    for i, (aid, a) in enumerate(items):
        key = f"{prefix}:{aid}"
        done = key in unlocked
        with cols[i % cols_per_row]:
            icon_style = "" if done else "filter:grayscale(1) opacity(0.35);"
            bg     = "rgba(0,180,216,0.12)" if done else "rgba(128,128,128,0.08)"
            border = "#00b4d8" if done else "rgba(150,150,150,0.35)"
            name_c = "#00b4d8" if done else "#8a94a6"
            desc_c = "#cbd5e1" if done else "#6b7280"
            badge  = "✅ 已解锁" if done else "🔒 未解锁"
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:14px;
                        padding:14px; margin-bottom:10px; display:flex; align-items:center; gap:12px;">
                <div style="font-size:34px; {icon_style}">{a['icon']}</div>
                <div style="flex:1;">
                    <div style="font-weight:800; font-size:15px; color:{name_c};">{a['name']}</div>
                    <div style="font-size:12px; color:{desc_c}; margin-top:2px;">{html.escape(a['desc'])}</div>
                    <div style="font-size:11px; color:{name_c}; margin-top:4px;">{badge}</div>
                </div>
            </div>""", unsafe_allow_html=True)

def render_sidebar_achievements(case_id: str):
    st.markdown("### 🏅 成就进度")
    total = achievement_total()
    got = achievement_unlocked_count()
    st.progress(min(1.0, got / total) if total else 0)
    st.caption(f"已解锁 **{got} / {total}**")
    if case_id in CASE_ACHIEVEMENTS:
        st.caption("本病例成就：")
        cols = st.columns(4)
        unlocked = st.session_state.get("achievements", [])
        for i, (aid, a) in enumerate(CASE_ACHIEVEMENTS[case_id].items()):
            done = f"{case_id}:{aid}" in unlocked
            style = "" if done else "filter:grayscale(1) opacity(0.35);"
            cols[i].markdown(
                f'<div title="{html.escape(a["name"])}：{html.escape(a["desc"])}" '
                f'style="text-align:center; font-size:24px; {style}">{a["icon"]}</div>',
                unsafe_allow_html=True)

# ================= 8. 开始界面 =================
if st.session_state.screen == "start":
    st.markdown('<div style="text-align: center; font-size: 72px; font-weight: 900; color: #00b4d8; margin-top: 80px; letter-spacing: 5px;">🏥 PediMind</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 38px; font-weight: 700; color: #e63946; margin-bottom: 30px; letter-spacing: 5px;">急诊风云</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; max-width: 600px; margin: 0 auto 50px auto; line-height: 2; font-size: 16px;">
        <span style="color: #00b4d8; font-weight: bold;">欢迎来到 PediMind 儿科急诊模拟器。</span><br>
        在这里，你是一名急诊科医生。<br>面对各种凶险的儿科急症，你的每一次决策，都将决定患儿的生死。
    </div>
    """, unsafe_allow_html=True)
    
    _, col_mid, _ = st.columns([1, 1, 1])
    with col_mid:
        st.text_input("👨‍⚕️ 你的昵称", key="player_name", max_chars=12)
        st.caption(f"专属 ID：`{st.session_state.player_id}`（收藏链接即可跨设备恢复）")
        if st.button("🚀 开始游戏", use_container_width=True, type="primary"):
            st.session_state.screen = "case_select"; st.rerun()
        
        st.info("💡 提示：数据与成就中心已移至左侧边栏，点击屏幕左上角的 `>` 即可展开查看。")
    st.stop()

# ================= 9. 病例选择 =================
if st.session_state.screen == "case_select":
    st.markdown('<div style="text-align: center; font-size: 42px; font-weight: 900; color: #00b4d8; margin-top: 30px; margin-bottom: 30px;">📋 选择病例</div>', unsafe_allow_html=True)
    for case_id, case in CASES_CONFIG.items():
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"""
            <div class="case-card">
                <div style="font-size: 24px; font-weight: bold; color: #00b4d8; margin-bottom: 10px;">📁 {case['name']} ({case['disease']})</div>
                <div style="font-size: 14px; margin-bottom: 5px;"><b>患儿：</b>{case['patient']}</div>
                <div style="font-size: 14px; color: #e63946; margin-bottom: 5px;"><b>难度：</b>{case['difficulty']}</div>
                <div style="font-size: 13px; font-style: italic; margin-top: 10px;">{case['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("▶️ 进入病例", key=f"select_{case_id}", use_container_width=True, type="primary"):
                st.session_state.current_case = case_id
                st.session_state.screen = "difficulty_select"; st.rerun()
    st.stop()

# ================= 10. 难度选择 =================
if st.session_state.screen == "difficulty_select":
    case = CASES_CONFIG[st.session_state.current_case]
    st.markdown('<div style="text-align: center; font-size: 42px; font-weight: 900; color: #00b4d8; margin-top: 30px;">⚙️ 选择难度</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.info("🌱 **简单模式**\n- 5个行动点\n- 初始信任值40\n- 有详细提示和引导")
        if st.button("开始简单模式", use_container_width=True):
            st.session_state.game_mode = "easy"; st.session_state.screen = "game"; st.rerun()
    with col2:
        st.warning("🔥 **困难模式**\n- 3个行动点\n- 初始信任值20\n- 无提示，盲盒线索")
        if st.button("开始困难模式", use_container_width=True):
            st.session_state.game_mode = "hard"; st.session_state.screen = "game"; st.rerun()
    st.stop()

# ================= 11. 成就殿堂 =================
if st.session_state.screen == "achievements":
    st.markdown('<div style="text-align:center; font-size:42px; font-weight:900; color:#00b4d8; margin:20px 0;">🏅 成就殿堂</div>', unsafe_allow_html=True)
    total = achievement_total()
    got = achievement_unlocked_count()
    c1, c2, c3 = st.columns(3)
    c1.metric("成就总数", total)
    c2.metric("已解锁", got)
    c3.metric("完成度", f"{(got/total*100 if total else 0):.0f}%")
    st.progress(min(1.0, got / total) if total else 0)
    st.divider()

    tab_names = ["🌐 通用成就"] + [f"📁 {CASES_CONFIG[cid]['disease']}" for cid in CASES_CONFIG]
    tabs = st.tabs(tab_names)
    with tabs[0]:
        st.caption("所有病例共享的通用成就")
        render_achievement_grid(GLOBAL_ACHIEVEMENTS, "global")
    for i, cid in enumerate(CASES_CONFIG, start=1):
        with tabs[i]:
            case = CASES_CONFIG[cid]
            st.caption(f"**{case['name']}** · {case['disease']} · {case['difficulty']}")
            render_achievement_grid(CASE_ACHIEVEMENTS.get(cid, {}), cid)
    st.stop()

# ================= 12. 错题本 =================
if st.session_state.screen == "profile":
    st.markdown('<div style="text-align:center; font-size:42px; font-weight:900; color:#00b4d8; margin:20px 0;">📕 我的错题本</div>', unsafe_allow_html=True)
    st.caption(f"👨‍⚕️ 当前玩家：**{st.session_state.player_name}**　｜ 专属 ID：`{st.session_state.player_id}`")
    records = fetch_my_records(st.session_state.player_id)
    if not records:
        st.info("还没有记录。完成一局游戏后，这里会自动收录你的失误与得分。")
    else:
        df = pd.DataFrame(records)
        c1, c2, c3 = st.columns(3)
        c1.metric("总场次", len(df))
        c2.metric("平均分", f"{df['final_score'].mean():.1f}")
        c3.metric("存活率", f"{df['survived'].mean()*100:.0f}%")
        st.subheader("📈 得分趋势")
        fig = px.line(df.iloc[::-1], x="created_at", y="final_score", markers=True, color="case_name")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("🗂️ 历史记录")
        for r in records:
            mistakes = json.loads(r["mistakes"] or "[]")
            with st.expander(f"{'✅' if r['survived'] else '💀'} {r['created_at']} | {r['disease']} | 得分 {r['final_score']}"):
                a, b = st.columns(2)
                a.write(f"**难度**：{r['mode']}\n\n**病情度**：{r['disease_progress']}\n\n"
                        f"**问诊/诊断/急救/沟通**：{r['score_inquiry']}/{r['score_diagnosis']}/{r['score_emergency']}/{r['score_empathy']}")
                with b:
                    if mistakes: st.warning("**失误记录：**\n" + "\n".join(f"- {m}" for m in mistakes))
                    else: st.success("本局无失误 🎉")
    st.stop()

# ================= 13. 全服统计 =================
if st.session_state.screen == "stats":
    st.markdown('<div style="text-align:center; font-size:42px; font-weight:900; color:#00b4d8; margin:20px 0;">📊 全服游玩统计</div>', unsafe_allow_html=True)
    all_recs = fetch_all_records()
    if not all_recs:
        st.info("暂无任何玩家数据。")
    else:
        df = pd.DataFrame(all_recs)
        c1, c2, c3 = st.columns(3)
        c1.metric("总游玩场次", len(df))
        c2.metric("独立玩家数", df["player_id"].nunique())
        c3.metric("全服平均分", f"{df['final_score'].mean():.1f}")
        st.subheader("🏥 各病例表现")
        agg = df.groupby("disease").agg(
            场次=("id", "count"), 平均分=("final_score", "mean"),
            最高分=("final_score", "max"), 存活率=("survived", "mean")).reset_index()
        agg["平均分"] = agg["平均分"].round(1)
        agg["存活率"] = (agg["存活率"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(agg, use_container_width=True, hide_index=True)
        st.subheader("📊 各病例平均分对比")
        fig2 = px.bar(agg, x="disease", y="平均分", color="平均分", color_continuous_scale="Teal", text="平均分")
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("🏅 成就解锁率（Top 10）")
        flat = flatten_achievements()
        total_rec = len(df)
        ach_stats = []
        for key, a in flat.items():
            cnt = sum(1 for x in df["achievements"] if key in json.loads(x or "[]"))
            if cnt > 0:
                ach_stats.append({"成就": f"{a['icon']} {a['name']}",
                                  "解锁率": f"{cnt/total_rec*100:.0f}%", "解锁人数": cnt})
        ach_stats.sort(key=lambda r: r["解锁人数"], reverse=True)
        st.dataframe(pd.DataFrame(ach_stats[:10]), use_container_width=True, hide_index=True)
    st.stop()

# ================= 14. 游戏主逻辑 =================
if st.session_state.screen == "game":
    case = CASES_CONFIG[st.session_state.current_case]
    mode = st.session_state.game_mode

    defaults = {
        "messages": [{"role": "assistant", "content": f"医生！您快看看我家孩子！{case['description']}"}],
        "trust_score": 40 if mode == "easy" else 20,
        "action_points": 5 if mode == "easy" else 3,
        "disease_progress": 30,
        "game_over": False, "time_period": 1, "unlocked_clues": [],
        "penalty_log": [], "crisis_correct_count": 0,
        "final_score": 0, "final_title": "",
        "score_inquiry": 0, "score_diagnosis": 0, "score_emergency": 0, "score_empathy": 0,
        "diagnosis_made": None, "diagnosis_processed": False,
        "auscultation_mode": False, "auscultation_completed": False,
        "physical_exam_done": {}, "comm_made": False,
        "crisis_start_time": time.time(), "record_saved": False,
        "intro_video_played": False
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    check_achievements(case, st.session_state.current_case)

    SCENARIO_DATA = {
        1: {"title": "🎬 第一幕：急诊室初遇", "mode": "free"},
        2: {"title": "🎬 第二幕：初步判断", "mode": "decision"},
        3: {"title": "🎬 第三幕：迷雾重重", "mode": "free"},
        4: {"title": "🎬 第四幕：生死时速", "mode": "crisis"},
        5: {"title": "🎬 第五幕：医患沟通", "mode": "communication"},
        6: {"title": "🎬 第六幕：结案复盘", "mode": "end"}
    }

    # 在侧边栏渲染当前状态
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### 🩺 当前：{case['disease']}")
        render_sidebar_achievements(st.session_state.current_case)

    # 开场视频
    if st.session_state.time_period == 1 and not st.session_state.intro_video_played:
        st.subheader("🎬 急诊室画面")
        if os.path.exists("static/intro_er.mp4"):
            st.video("static/intro_er.mp4", autoplay=True, muted=True)
        else:
            st.info("（此处为开场视频：深夜急诊，一位母亲抱着孩子冲进抢救室…）")
        if st.button("⏭️ 跳过开场"):
            st.session_state.intro_video_played = True; st.rerun()
        st.session_state.intro_video_played = True

    st.markdown(f'<div style="display: flex; justify-content: space-around; gap: 15px; margin-bottom: 25px;"><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">❤️ 信任值</div><div style="font-size:20px; font-weight:800; color:#00b4d8;">{st.session_state.trust_score}</div></div><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">⏳ 行动点</div><div style="font-size:20px; font-weight:800; color:#f6e05e;">{st.session_state.action_points}</div></div><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">⚠️ 病情度</div><div style="font-size:20px; font-weight:800; color:#e63946;">{st.session_state.disease_progress}</div></div></div>', unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2.5, 1.2])

    with col_left:
        st.subheader("📈 生命体征")
        dp = st.session_state.disease_progress
        render_monitor(dp)
        if dp < 40: st.caption("🧠 意识：轻度烦躁")
        elif dp < 70: st.caption("🧠 意识：明显烦躁")
        elif dp < 90: st.caption("🧠 意识：发绀、烦躁")
        else: st.caption("🧠 意识：意识模糊")
        st.divider()
        if not st.session_state.game_over:
            if st.session_state.time_period in [1, 3]:
                if st.session_state.time_period == 3 and st.session_state.action_points <= 2 and not st.session_state.diagnosis_processed:
                    st.warning("⚠️ 请给出初步诊断：")
                    for opt_key, opt_val in case["diagnosis"].items():
                        if st.button(opt_val, key=f"diag_{opt_key}", use_container_width=True):
                            st.session_state.diagnosis_processed = True
                            st.session_state.diagnosis_made = opt_key
                            if opt_val == case["disease"]:
                                st.session_state.score_diagnosis += 15
                                st.session_state.messages.append({"role": "assistant", "content": f"✅ 诊断正确！这是{case['disease']}。"})
                            else:
                                st.session_state.disease_progress = min(100, st.session_state.disease_progress + 25)
                                st.session_state.penalty_log.append(f"误诊为{opt_val}")
                                st.session_state.messages.append({"role": "assistant", "content": "❌ 误诊！病情加重。"})
                            st.rerun()
                elif st.session_state.action_points <= 2 and (st.session_state.time_period == 1 or st.session_state.diagnosis_processed):
                    if st.button("▶️ 进入下一幕", use_container_width=True):
                        st.session_state.time_period += 1
                        st.session_state.action_points = 3 if st.session_state.time_period == 4 else 5
                        st.session_state.disease_progress += 10; st.rerun()
                elif st.session_state.action_points > 2:
                    st.caption(f"💡 强制问诊：还需 {st.session_state.action_points - 2} 次")
            if st.session_state.time_period == 3 and st.session_state.diagnosis_processed and st.session_state.action_points <= 2:
                if st.button("🚨 突发事件！进入第四幕", use_container_width=True):
                    st.session_state.time_period = 4; st.session_state.action_points = 3
                    st.session_state.disease_progress = max(70, st.session_state.disease_progress + 15)
                    st.session_state.crisis_start_time = time.time()
                    st.session_state.messages.append({"role": "assistant", "content": "（患儿突然剧烈哭闹，呼吸困难急剧加重）医生！救救她！"}); st.rerun()
            if st.session_state.time_period == 4 and st.session_state.crisis_correct_count >= 2:
                if st.button("✅ 进入第五幕", use_container_width=True):
                    st.session_state.score_emergency += 20; st.session_state.time_period = 5
                    st.session_state.action_points = 5
                    st.session_state.messages.append({"role": "assistant", "content": "（经过处理，患儿呼吸平稳）"}); st.rerun()
            if st.session_state.time_period == 5 and not st.session_state.comm_made:
                st.warning("⚠️ 交代病情：")
                if st.button("A. 赶紧办住院", key="c_a"):
                    st.session_state.comm_made = True; st.session_state.trust_score -= 10; st.rerun()
                if st.button("B. 病情很重，可能窒息", key="c_b"):
                    st.session_state.comm_made = True; st.session_state.trust_score -= 5; st.rerun()
                if st.button("C. 您别自责，送来及时，我们会尽力", key="c_c"):
                    st.session_state.comm_made = True; st.session_state.trust_score += 15
                    st.session_state.score_empathy += 15; st.rerun()
            if st.session_state.time_period == 5 and st.session_state.comm_made:
                if st.button("▶️ 进入复盘", use_container_width=True):
                    st.session_state.time_period = 6; st.rerun()
        st.divider()
        if st.session_state.time_period >= 5 and not st.session_state.game_over:
            if st.button("📝 提交诊断，结束", use_container_width=True):
                st.session_state.game_over = True; st.rerun()
        if st.button("🔄 重新开始", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "screen", "current_case", "game_mode",
                               "player_id", "player_name", "achievements", "completed_cases"]:
                    del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()

    with col_right:
        st.subheader("🔍 线索夹")
        if mode == "hard":
            for i, clue in enumerate(case["clues"]):
                if clue in st.session_state.unlocked_clues: st.success(f"✅ {clue}")
                else: st.text(f"🔒 未知线索 {i+1}/{len(case['clues'])}")
        else:
            for clue in case["clues"]:
                if clue in st.session_state.unlocked_clues: st.success(f"✅ {clue}")
                else: st.text("🔒 未知线索")
        if st.session_state.penalty_log:
            st.divider(); st.caption("📝 失误记录")
            for log in st.session_state.penalty_log: st.warning(log)

    with col_center:
        if not st.session_state.game_over:
            st.title(SCENARIO_DATA[st.session_state.time_period]["title"])
            if st.session_state.time_period == 1: st.info("任务：安抚情绪，了解发病时间和特征。")
        else: st.title("🏥 结案复盘")

        if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
            with st.expander("🩺 体格检查工具箱（消耗行动点）", expanded=False):
                tab1, tab2, tab3 = st.tabs(["👁️ 视诊", "👂 听诊", "🖐️ 触诊/叩诊"])
                
                with tab1:
                    if st.button("进行视诊", use_container_width=True):
                        if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("vis"):
                            st.session_state.action_points -= 1
                            st.session_state.physical_exam_done["vis"] = True
                            st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                            
                            if st.session_state.current_case == "laryngitis":
                                st.session_state.unlocked_clues.append("三凹征")
                                st.session_state.messages.append({"role": "assistant", "content": "（视诊：患儿吸气时胸骨上窝、锁骨上窝及肋间隙明显凹陷，即“三凹征”。）"})
                            elif st.session_state.current_case == "foreign_body":
                                st.session_state.unlocked_clues.append("面色发绀")
                                st.session_state.messages.append({"role": "assistant", "content": "（视诊：患儿面色口唇发绀，剧烈咳嗽，双手抓喉。）"})
                            elif st.session_state.current_case == "asthma":
                                st.session_state.unlocked_clues.append("呼气性呼吸困难")
                                st.session_state.messages.append({"role": "assistant", "content": "（视诊：患儿端坐呼吸，呼气相延长，大汗淋漓。）"})
                            elif st.session_state.current_case == "epiglottitis":
                                st.session_state.unlocked_clues.extend(["流涎", "拒绝平卧"])
                                st.session_state.messages.append({"role": "assistant", "content": "（视诊：患儿呈端坐位，流涎明显，拒绝平卧，精神极差。）"})
                            st.rerun()
                        elif st.session_state.action_points <= 0:
                            st.error("行动点不足！")
                            
                with tab2:
                    if st.button("进行听诊", use_container_width=True):
                        if st.session_state.action_points > 0 and not st.session_state.auscultation_completed:
                            st.session_state.action_points -= 1
                            st.session_state.auscultation_mode = True; st.rerun()
                        elif st.session_state.action_points <= 0:
                            st.error("行动点不足！")
                            
                with tab3:
                    if st.button("进行触诊/叩诊", use_container_width=True):
                        if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("palp"):
                            st.session_state.action_points -= 1
                            st.session_state.physical_exam_done["palp"] = True
                            st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                            
                            if st.session_state.current_case == "laryngitis":
                                st.session_state.messages.append({"role": "assistant", "content": "（触诊：颈部无明显红肿，双肺呼吸音粗。叩诊：清音。）"})
                            elif st.session_state.current_case == "foreign_body":
                                st.session_state.unlocked_clues.append("呼吸音不对称")
                                st.session_state.messages.append({"role": "assistant", "content": "（触诊：气管居中，语颤减弱。叩诊：右侧呈过清音，提示异物阻塞。）"})
                            elif st.session_state.current_case == "asthma":
                                st.session_state.messages.append({"role": "assistant", "content": "（触诊：语颤减弱。叩诊：双肺过清音，肺界下移。）"})
                            elif st.session_state.current_case == "epiglottitis":
                                st.session_state.messages.append({"role": "assistant", "content": "（触诊：颈前区拒按，下颌淋巴结肿大。注意：严禁强行压舌检查！）"})
                            st.rerun()
                        elif st.session_state.action_points <= 0:
                            st.error("行动点不足！")

        if st.session_state.auscultation_mode:
            st.warning("🩺 听诊音频：")
            if os.path.exists(f"static/{case['audio']}"): st.audio(f"static/{case['audio']}", format="audio/mp3")
            st.write("请判断：")
            if st.button("吸气性喉鸣 (Stridor)"):
                if case["disease"] in ("急性感染性喉炎", "急性会厌炎"):
                    st.session_state.unlocked_clues.append("吸气性喉鸣")
                    st.session_state.score_diagnosis += 10
                st.session_state.auscultation_mode = False
                st.session_state.auscultation_completed = True; st.rerun()
            if st.button("呼气性哮鸣音 (Wheezing)"):
                if case["disease"] == "支气管哮喘急性发作":
                    st.session_state.unlocked_clues.append("双肺哮鸣音")
                    st.session_state.score_diagnosis += 10
                st.session_state.auscultation_mode = False
                st.session_state.auscultation_completed = True; st.rerun()
        else:
            with st.container(height=400):
                for msg in st.session_state.messages:
                    render_message(msg["role"], msg["content"])

            if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "crisis":
                elapsed = time.time() - st.session_state.crisis_start_time
                time_left = max(0, 90 - int(elapsed))
                st.error(f"⏰ 紧急抢救倒计时：{time_left} 秒！请选择：")
                if time_left <= 0:
                    st.session_state.disease_progress = 100; st.session_state.game_over = True; st.rerun()
                if st.session_state.action_points > 0:
                    for opt_key, opt_val in case["crisis"].items():
                        if st.button(opt_val["label"], key=f"crisis_{opt_key}", use_container_width=True):
                            st.session_state.action_points -= 1
                            st.session_state.messages.append({"role": "user", "content": opt_val["label"]})
                            st.session_state.messages.append({"role": "assistant", "content": opt_val["reply"]})
                            st.session_state.disease_progress += opt_val["disease"]
                            if opt_val["type"] == "correct": st.session_state.crisis_correct_count += 1
                            else: st.session_state.penalty_log.append(opt_val["label"])
                            if st.session_state.disease_progress >= 100: st.session_state.game_over = True
                            st.rerun()

            elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
                if prompt := st.chat_input("输入问诊或检查..."):
                    if st.session_state.action_points <= 0:
                        st.warning("行动点已用完，请进入下一幕。")
                    else:
                        st.session_state.action_points -= 1
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        
                        # 线索匹配
                        for clue in case["clues"]:
                            if clue in prompt or (clue == "犬吠样咳嗽" and "小狗" in prompt) or (clue == "进食呛咳史" and "花生" in prompt):
                                if clue not in st.session_state.unlocked_clues:
                                    st.session_state.unlocked_clues.append(clue)
                                    st.session_state.score_inquiry += 8
                                    st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)

                        # 大模型生成回复
                        try:
                            api_messages = [
                                {"role": "system", "content": f"你正在扮演：{case['ai_role']}。规则：{case['ai_rules']}。当前医生对你说了：{prompt}"},
                                {"role": "user", "content": prompt}
                            ]
                            response = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=api_messages,
                                temperature=0.7,
                                max_tokens=100
                            )
                            ai_reply = response.choices[0].message.content
                        except Exception as e:
                            # 兜底回复（纯白话语境，无提示词）
                            if st.session_state.current_case == "laryngitis":
                                ai_reply = "医生，她昨晚就开始咳嗽，声音像小狗叫一样，刚才突然喘不上气，我该怎么办啊！"
                            elif st.session_state.current_case == "foreign_body":
                                ai_reply = "医生！她刚才吃花生突然咳得整个人弓起来，脸都憋紫了，求您快救救她！"
                            elif st.session_state.current_case == "asthma":
                                ai_reply = "医生，她以前也喘过，但这次特别厉害，呼气很费力，您快帮帮她。"
                            elif st.session_state.current_case == "epiglottitis":
                                ai_reply = "医生，她一直流口水，一躺下就喘不上气，还发高烧，快救救孩子吧！"
                            else:
                                ai_reply = "医生，我真的好担心，求您快帮帮孩子吧！"

                        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                        st.rerun()

    if st.session_state.game_over:
        st.divider(); st.header("🩺 带教老师复盘")
        if st.session_state.final_score == 0:
            total = (st.session_state.score_inquiry + st.session_state.score_diagnosis
                     + st.session_state.score_emergency + st.session_state.score_empathy)
            st.session_state.final_score = total
            if total >= 80: st.session_state.final_title = "🏆 儿科急诊之光"
            elif total >= 60: st.session_state.final_title = "🌟 有潜力的住院医"
            else: st.session_state.final_title = "📚 还需回炉重造"

        if not st.session_state.record_saved:
            check_achievements(case, st.session_state.current_case)
            save_record({
                "player_id": st.session_state.player_id,
                "player_name": st.session_state.player_name,
                "case_id": st.session_state.current_case,
                "case_name": case["name"],
                "disease": case["disease"],
                "mode": st.session_state.game_mode,
                "final_score": st.session_state.final_score,
                "score_inquiry": st.session_state.score_inquiry,
                "score_diagnosis": st.session_state.score_diagnosis,
                "score_emergency": st.session_state.score_emergency,
                "score_empathy": st.session_state.score_empathy,
                "disease_progress": st.session_state.disease_progress,
                "survived": 1 if st.session_state.disease_progress < 100 else 0,
                "mistakes": json.dumps(st.session_state.penalty_log, ensure_ascii=False),
                "achievements": json.dumps(st.session_state.achievements, ensure_ascii=False),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.session_state.record_saved = True

        st.info(f"### {st.session_state.final_title}\n最终得分：{st.session_state.final_score} / 100")
        st.subheader("📊 个人能力雷达图")
        fig = go.Figure(go.Scatterpolar(
            r=[min(100, st.session_state.score_inquiry * 3),
               min(100, st.session_state.score_diagnosis * 2),
               min(100, st.session_state.score_emergency * 2),
               min(100, st.session_state.score_empathy * 2)],
            theta=['问诊', '诊断', '急救', '沟通'], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏅 成就总览")
        st.caption(f"已解锁 **{achievement_unlocked_count()} / {achievement_total()}** · 详情请到左侧边栏「成就殿堂」查看")
        
        if st.button("🏠 返回主菜单"):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "player_id", "player_name", "achievements", "completed_cases"]:
                    del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()
