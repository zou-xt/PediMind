import streamlit as st
import streamlit.components.v1 as components
import os, time, json, base64, html, sqlite3, uuid, re, io
from datetime import datetime
from contextlib import contextmanager
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import qrcode

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

# ================= 2. 四大预设病例 (无剧透版) =================
CASES_CONFIG = {
    "laryngitis": {
        "name": "2岁患儿的犬吠声", "disease": "急性感染性喉炎", "patient": "小雨，2岁3个月，女", "difficulty": "⭐⭐⭐⭐",
        "description": "深夜急诊，患儿突发呼吸困难，家长极度焦虑。",
        "opening_line": "医生！您快看看我家孩子！突然就这样了，我快吓死了！",
        "avatar": "static/mom.png",
        "audio": "stridor.mp3",
        "clues": ["犬吠样咳嗽", "夜间加重", "吸气性喉鸣", "白天感冒史", "三凹征"],
        "ai_role": "2岁急性喉炎患儿的妈妈。极度焦虑、自责。",
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
        "opening_line": "医生！快救救他！刚才还好好的，突然就这样了！",
        "avatar": "static/mom.png",
        "audio": "wheezing.mp3",
        "clues": ["进食呛咳史", "剧烈呛咳", "呼吸音不对称", "面色发绀"],
        "ai_role": "1岁气道异物患儿的妈妈。极度恐慌、语无伦次。",
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
        "opening_line": "医生，她突然喘得很厉害，您快帮帮她！",
        "avatar": "static/mom.png",
        "audio": "wheezing.mp3",
        "clues": ["既往哮喘史", "过敏史", "呼气性呼吸困难", "双肺哮鸣音"],
        "ai_role": "4岁哮喘患儿的妈妈。疲惫、担忧。",
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
        "opening_line": "医生，孩子发高烧，现在状态特别差，快救救她吧！",
        "avatar": "static/mom.png",
        "audio": "stridor.mp3",
        "clues": ["高热", "流涎", "吞咽困难", "拒绝平卧", "无犬吠样咳嗽"],
        "ai_role": "3岁会厌炎患儿的妈妈。极度恐惧。",
        "diagnosis": {"A": "急性感染性喉炎", "B": "急性会厌炎", "C": "气道异物", "D": "支气管哮喘"},
        "crisis": {
            "A": {"label": "保持坐位，吸氧，静脉使用糖皮质激素，准备气管切开/插管", "type": "correct", "disease": -30, "reply": "（气道水肿减轻，准备进一步气道管理）"},
            "B": {"label": "强行使用压舌板检查咽喉部", "type": "harmful", "disease": 50, "reply": "（诱发喉痉挛，呼吸骤停！）"},
            "C": {"label": "平卧位，给予镇静剂", "type": "harmful", "disease": 40, "reply": "（平卧+镇静剂，迅速导致窒息）"},
            "D": {"label": "等待放射科拍颈部侧位片", "type": "invalid", "disease": 20, "reply": "（等待过程中气道完全梗阻）"}
        }
    }
}

# ================= 2.5 线索同义词及意图库 =================
CLUE_KEYWORDS = {
    "犬吠样咳嗽": ["小狗叫", "狗叫", "空空", "犬吠", "像小狗"],
    "夜间加重": ["晚上", "半夜", "夜里", "凌晨"],
    "吸气性喉鸣": ["吸气", "喉咙响", "拉风箱", "吸不进去"],
    "白天感冒史": ["感冒", "流鼻涕", "白天还好"],
    "三凹征": ["胸口凹", "凹陷", "锁骨", "肋间", "脖子下面凹"],
    "进食呛咳史": ["吃东西", "花生", "呛", "卡住"],
    "剧烈呛咳": ["剧烈咳嗽", "咳得", "拼命咳"],
    "呼吸音不对称": ["一边没声", "不对称", "右边"],
    "面色发绀": ["脸紫", "嘴唇紫", "发紫", "憋紫"],
    "既往哮喘史": ["以前也喘", "小时候喘", "哮喘", "喘过"],
    "过敏史": ["过敏", "花粉", "起疹子"],
    "呼气性呼吸困难": ["呼气", "吐气", "出气", "吹气"],
    "双肺哮鸣音": ["嘶嘶", "拉风箱", "声音尖"],
    "高热": ["发烧", "烫", "高烧"],
    "流涎": ["流口水", "口水", "咽不下去"],
    "吞咽困难": ["不想吃", "吞不下", "一吃就哭"],
    "拒绝平卧": ["躺下", "一躺", "平躺", "坐下"],
}

CLUE_EXPLANATIONS = {
    "犬吠样咳嗽": "声门下水肿导致，咳嗽声音如犬吠。高度提示急性喉炎。",
    "夜间加重": "喉部迷走神经张力夜间增高，导致气道痉挛加重，是喉炎典型特征。",
    "吸气性喉鸣": "上气道梗阻的标志。吸气时气流通过狭窄的声门产生。",
    "三凹征": "胸骨上窝、锁骨上窝、肋间隙凹陷。提示极度呼吸困难，需紧急干预！",
    "进食呛咳史": "异物吸入最常见的主诉。由于婴幼儿磨牙未全，极易误吸坚果类。",
    "面色发绀": "血氧饱和度下降至 90% 以下的表现，提示严重缺氧。",
    "既往哮喘史": "气道高反应性，容易在接触过敏原或感染后诱发急性喘息。",
    "呼气性呼吸困难": "下气道梗阻表现，呼气时相延长，见于哮喘或毛细支气管炎。",
    "高热": "体温 > 39°C。会厌炎常伴有高热和吞咽困难。",
    "流涎": "因吞咽疼痛或困难导致，是会厌炎的高危信号，绝对禁止平卧！",
    "拒绝平卧": "会厌炎患儿为了保持气道开放，被迫采取端坐位。强行平卧可致窒息。",
}

def check_clue_trigger(clue: str, prompt: str) -> bool:
    p = prompt.lower()
    if clue in CLUE_KEYWORDS:
        if any(kw in p for kw in CLUE_KEYWORDS[clue]): return True
    if clue == "犬吠样咳嗽" and ("咳" in p or "小狗" in p or "空空" in p): return True
    elif clue == "夜间加重" and any(k in p for k in ["晚上", "夜里", "半夜"]): return True
    elif clue == "三凹征" and any(k in p for k in ["喘", "呼吸", "凹", "陷"]): return True
    elif clue == "进食呛咳史" and any(k in p for k in ["吃", "呛", "花生", "异物"]): return True
    elif clue == "既往哮喘史" and any(k in p for k in ["以前", "既往", "喘过"]): return True
    return False

def check_medical_dispute():
    ss = st.session_state
    if ss.game_over: return False
    if ss.trust_score <= 0:
        ss.game_over = True
        ss.disease_progress = 100
        ss.penalty_log.append("医患冲突爆发，延误治疗致死")
        ss.messages.append({"role": "assistant", "content": "（家属彻底失去理智，在急诊室大闹起来，阻拦你的任何抢救操作。保安介入，上级医生接手，但患儿已错过最佳抢救时机……）"})
        return True
    return False

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match: return match.group(0)
    return text

# ================= 3. 成就系统 =================
GLOBAL_ACHIEVEMENTS = {
    "first_inquiry": {"icon": "🩺", "name": "初次问诊", "desc": "迈出问诊第一步", "check": lambda ss, c: any(m.get("role") == "user" for m in getattr(ss, "messages", []))},
    "perfect_score": {"icon": "🏆", "name": "急诊之光", "desc": "单局得分 ≥ 80", "check": lambda ss, c: getattr(ss, "final_score", 0) >= 80},
    "all_rounder":   {"icon": "📚", "name": "全能住院医", "desc": "通关全部 4 个病例", "check": lambda ss, c: len(getattr(ss, "completed_cases", set())) >= 4},
    "empathy":       {"icon": "💬", "name": "医者仁心", "desc": "用共情方式交代病情", "check": lambda ss, c: getattr(ss, "score_empathy", 0) > 0},
}

CASE_ACHIEVEMENTS = {
    "laryngitis": {
        "bark_cough":   {"icon": "🐕", "name": "犬吠辨识者", "desc": "识别出「犬吠样咳嗽」", "check": lambda ss, c: "犬吠样咳嗽" in getattr(ss, "unlocked_clues", [])},
        "calm_parent":  {"icon": "🤝", "name": "安抚专家",   "desc": "信任值提升至 ≥ 50", "check": lambda ss, c: getattr(ss, "trust_score", 0) >= 50},
    },
    "foreign_body": {
        "peanut_detective": {"icon": "🥜", "name": "花生侦探", "desc": "识别出「进食呛咳史」", "check": lambda ss, c: "进食呛咳史" in getattr(ss, "unlocked_clues", [])},
        "heimlich_master":  {"icon": "🤲", "name": "海姆立克大师","desc": "急救处置正确 ≥ 2 次", "check": lambda ss, c: getattr(ss, "crisis_correct_count", 0) >= 2},
    },
    "asthma": {
        "history_hunter": {"icon": "📋", "name": "病史侦探", "desc": "识别出「既往哮喘史」", "check": lambda ss, c: "既往哮喘史" in getattr(ss, "unlocked_clues", [])},
        "steady_breath":  {"icon": "🫁", "name": "呼吸守护", "desc": "以病情度 < 50 通关", "check": lambda ss, c: getattr(ss, "game_over", False) and getattr(ss, "disease_progress", 100) < 50},
    },
    "epiglottitis": {
        "drool_watcher": {"icon": "💧", "name": "流涎观察员", "desc": "识别出「流涎」", "check": lambda ss, c: "流涎" in getattr(ss, "unlocked_clues", [])},
        "no_flat":       {"icon": "🚫", "name": "拒绝平卧",   "desc": "识别出「拒绝平卧」", "check": lambda ss, c: "拒绝平卧" in getattr(ss, "unlocked_clues", [])},
    },
}

def flatten_achievements():
    flat = {f"global:{k}": v for k, v in GLOBAL_ACHIEVEMENTS.items()}
    for cid, achs in CASE_ACHIEVEMENTS.items():
        for k, v in achs.items():
            flat[f"{cid}:{k}"] = v
    return flat

# ================= 4. 数据库 =================
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
            id INTEGER PRIMARY KEY AUTOINCREMENT, player_id TEXT, player_name TEXT,
            case_id TEXT, case_name TEXT, disease TEXT, mode TEXT, final_score INTEGER,
            score_inquiry INTEGER, score_diagnosis INTEGER, score_emergency INTEGER, score_empathy INTEGER,
            disease_progress INTEGER, survived INTEGER, mistakes TEXT, achievements TEXT, created_at TEXT
        )""")
init_db()

def save_record(rec: dict):
    with db() as c:
        c.execute("""INSERT INTO records
            (player_id,player_name,case_id,case_name,disease,mode,final_score,score_inquiry,score_diagnosis,score_emergency,score_empathy,disease_progress,survived,mistakes,achievements,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec["player_id"], rec["player_name"], rec["case_id"], rec["case_name"], rec["disease"], rec["mode"], rec["final_score"], rec["score_inquiry"], rec["score_diagnosis"], rec["score_emergency"], rec["score_empathy"], rec["disease_progress"], rec["survived"], rec["mistakes"], rec["achievements"], rec["created_at"]))

def fetch_my_records(player_id: str):
    with db() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("SELECT * FROM records WHERE player_id=? ORDER BY id DESC", (player_id,)).fetchall()]

def fetch_all_records():
    with db() as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("SELECT * FROM records ORDER BY id DESC").fetchall()]

# ================= 5. 页面与高级 UI 初始化 =================
st.set_page_config(page_title="PediMind · 急诊风云", page_icon="🏥", layout="wide")

if "theme" not in st.session_state: st.session_state.theme = "day"
if "screen" not in st.session_state: st.session_state.screen = "start"
if "current_case" not in st.session_state: st.session_state.current_case = None
if "game_mode" not in st.session_state: st.session_state.game_mode = None
if "player_name" not in st.session_state: st.session_state.player_name = "实习医生"
if "achievements" not in st.session_state: st.session_state.achievements = []
if "completed_cases" not in st.session_state: st.session_state.completed_cases = set()

if "player_id" not in st.session_state:
    pid = st.query_params.get("pid")
    if pid and pid.strip(): st.session_state.player_id = pid.strip()
    else:
        st.session_state.player_id = str(uuid.uuid4())[:8]
        st.query_params["pid"] = st.session_state.player_id

# 🌟 极致 UI 质感 CSS
FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Noto+Sans+SC:wght@400;500;700;900&display=swap');"

CHAT_SPECIFIC_CSS = """
<style>
.stButton > button { font-size: 13px !important; padding: 4px 8px !important; border-radius: 20px !important; }
div[data-testid="stFormSubmitButton"] > button { border-radius: 8px !important; background-color: #0ea5e9 !important; color: white !important; border: none !important; height: 100%; }
div[data-testid="stFormSubmitButton"] > button:hover { background-color: #0284c7 !important; color: white !important; }
</style>
"""

if st.session_state.theme == "night":
    theme_css = f"""<style>
        {FONT_IMPORT}
        .stApp {{ background: #0a0f18 !important; color: #e2e8f0 !important; }}
        p, label, h1, h2, h3, h4, h5, h6 {{ color: #e2e8f0 !important; font-family: 'Inter', 'Noto Sans SC', sans-serif !important; letter-spacing: 0.5px; }}
        section[data-testid="stSidebar"] {{ background: rgba(13, 27, 42, 0.95) !important; backdrop-filter: blur(20px); border-right: 1px solid rgba(255,255,255,0.05); }}
        
        .stButton>button {{ border-radius: 10px !important; border: 1px solid rgba(14, 165, 233, 0.3) !important; background: rgba(14, 165, 233, 0.1) !important; color: #e2e8f0 !important; transition: all 0.2s ease; }}
        .stButton>button:hover {{ background: rgba(14, 165, 233, 0.25) !important; transform: translateY(-1px); }}
        
        .status-badge {{ background: rgba(255, 255, 255, 0.03); border-radius: 16px; padding: 12px 20px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); }}
        .case-card {{ padding: 30px; border-radius: 20px; margin-bottom: 20px; background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); transition: transform 0.2s; }}
        .case-card:hover {{ transform: translateY(-2px); border-color: rgba(14, 165, 233, 0.5); }}
        
        div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: rgba(255,255,255,0.08) !important; border-radius: 20px !important; padding: 20px !important; background: rgba(255,255,255,0.02) !important; backdrop-filter: blur(10px); }}
    </style>"""
else:
    theme_css = f"""<style>
        {FONT_IMPORT}
        .stApp {{ background: #f4f7fa !important; color: #1e293b !important; }}
        p, label, h1, h2, h3, h4, h5, h6 {{ color: #1e293b !important; font-family: 'Inter', 'Noto Sans SC', sans-serif !important; letter-spacing: 0.5px; }}
        section[data-testid="stSidebar"] {{ background: #ffffff !important; border-right: 1px solid #e2e8f0; }}
        
        .stButton>button {{ border-radius: 10px !important; border: 1px solid #cbd5e1 !important; background: #ffffff !important; color: #0f172a !important; font-weight: 600 !important; box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: all 0.2s ease; }}
        .stButton>button:hover {{ background: #f8fafc !important; border-color: #94a3b8 !important; transform: translateY(-1px); }}
        
        .status-badge {{ background: #ffffff; border-radius: 16px; padding: 15px 20px; text-align: center; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02); }}
        .case-card {{ padding: 30px; border-radius: 20px; margin-bottom: 20px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05); transition: transform 0.2s, box-shadow 0.2s; }}
        .case-card:hover {{ transform: translateY(-2px); box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); }}
        
        div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: #e2e8f0 !important; border-radius: 24px !important; padding: 20px !important; background: #ffffff !important; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.05); }}
    </style>"""

st.markdown(theme_css, unsafe_allow_html=True)
st.markdown(CHAT_SPECIFIC_CSS, unsafe_allow_html=True)

# ================= 侧边栏 =================
with st.sidebar:
    st.markdown("### 🎮 系统菜单")
    if st.session_state.screen != "start":
        if st.button("🏠 返回主菜单", use_container_width=True, type="primary"):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "player_id", "player_name", "achievements", "completed_cases"]: del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()
    st.markdown("---")
    st.markdown("### 📊 数据与成就")
    if st.button("📕 我的错题本", use_container_width=True): st.session_state.screen = "profile"; st.rerun()
    if st.button("📊 全服统计", use_container_width=True): st.session_state.screen = "stats"; st.rerun()
    if st.button("🏅 成就殿堂", use_container_width=True): st.session_state.screen = "achievements"; st.rerun()
    st.markdown("---")
    st.markdown("### 🎨 主题设置")
    if st.button("☀️ 白天模式" if st.session_state.theme == "night" else "🌙 夜晚模式", use_container_width=True):
        st.session_state.theme = "day" if st.session_state.theme == "night" else "night"
        st.rerun()
    with st.expander("🔑 换设备？恢复存档"):
        recover = st.text_input("专属 ID", max_chars=8, key="recover_id")
        if st.button("恢复", key="recover_btn") and recover.strip():
            st.session_state.player_id = recover.strip()
            st.query_params["pid"] = recover.strip()
            st.success(f"✅ 已切换"); st.rerun()

# ================= 6. 辅助函数 =================
@st.cache_data(show_spinner=False)
def load_avatar(path: str) -> str:
    if not os.path.exists(path): return ""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f: b64 = base64.b64encode(f.read()).decode()
    return f"data:image/{mime};base64,{b64}"

def render_message(role: str, content: str, case: dict = None):
    safe = html.escape(str(content)).replace("\n", "<br>")
    night = st.session_state.theme == "night"
    AVATAR_SIZE = "38px"
    def avatar_html(img_path, fallback_emoji, fallback_bg):
        src = load_avatar(img_path)
        if src: return (f'<img src="{src}" style="width:{AVATAR_SIZE}; height:{AVATAR_SIZE}; border-radius:50%; object-fit:cover; flex-shrink:0; box-shadow:0 1px 3px rgba(0,0,0,0.2);">')
        return (f'<div style="width:{AVATAR_SIZE}; height:{AVATAR_SIZE}; border-radius:50%; background:{fallback_bg}; color:#fff; display:flex; align-items:center; justify-content:center; flex-shrink:0; font-size:18px;">{fallback_emoji}</div>')
    
    if role == "user":
        bg = "#10b981" if not night else "#059669"; fg = "#ffffff"
        av = avatar_html("static/doctor.png", "👨‍⚕️", "#0ea5e9")
        st.markdown(f'<div style="display:flex; justify-content:flex-end; align-items:flex-start; margin:12px 0;"><div style="background:{bg}; color:{fg}; padding:12px 16px; border-radius:16px; border-top-right-radius:4px; max-width:75%; word-break:break-word; font-size:15px; line-height:1.6; box-shadow:0 2px 4px rgba(0,0,0,0.05);">{safe}</div><div style="margin-left:10px;">{av}</div></div>', unsafe_allow_html=True)
    else:
        bg = "#ffffff" if not night else "rgba(255,255,255,0.08)"; fg = "#1e293b" if not night else "#e2e8f0"
        patient_avatar = case.get("avatar", "static/mom.png") if case else "static/mom.png"
        av = avatar_html(patient_avatar, "👩", "#f59e0b")
        st.markdown(f'<div style="display:flex; justify-content:flex-start; align-items:flex-start; margin:12px 0;"><div style="margin-right:10px;">{av}</div><div style="background:{bg}; color:{fg}; padding:12px 16px; border-radius:16px; border-top-left-radius:4px; max-width:75%; word-break:break-word; font-size:15px; line-height:1.6; box-shadow:0 2px 4px rgba(0,0,0,0.05); border: 1px solid {"rgba(255,255,255,0.05)" if night else "#e2e8f0"};">{safe}</div></div>', unsafe_allow_html=True)

def render_monitor(disease_progress: int):
    if disease_progress < 40: hr, spo2, rr, bp_sys, bp_dia, status = 120, 97, 28, 100, 60, "normal"
    elif disease_progress < 70: hr, spo2, rr, bp_sys, bp_dia, status = 140, 93, 35, 90, 50, "warn"
    elif disease_progress < 90: hr, spo2, rr, bp_sys, bp_dia, status = 160, 88, 45, 80, 40, "danger"
    else: hr, spo2, rr, bp_sys, bp_dia, status = 180, 82, 55, 60, 30, "critical"

    html_code = f"""<html><head><style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      body {{ background:transparent; overflow:hidden; font-family:'Courier New',monospace; display:flex; justify-content:center; }}
      .monitor {{ position:relative; width:100%; max-width: 480px; height:280px; background:#000; border-radius:12px; border:1px solid #333; overflow:hidden; display:flex; box-shadow: inset 0 0 20px rgba(0,0,0,0.9); }}
      .wave-area {{ flex: 1; position: relative; background:#050505; min-width: 0; }} 
      canvas {{ display:block; width:100%; height:100%; }}
      .vitals {{ width: 120px; background: #000; padding: 12px 10px; display: flex; flex-direction: column; justify-content: space-between; border-left: 1px solid #222; flex-shrink: 0; }}
      .vital-item {{ display: flex; flex-direction: column; align-items: flex-start; line-height: 1; padding-bottom: 4px; }}
      .vital-label {{ font-size: 12px; font-weight: bold; margin-bottom: 2px; letter-spacing: 1px; }}
      .vital-value {{ font-size: 32px; font-weight: bold; font-family: 'Arial', sans-serif; line-height: 1.1; }}
      .vital-unit  {{ font-size: 10px; color: #666; margin-top: 2px; }}
      .c-green {{ color:#00ff00; text-shadow: 0 0 8px rgba(0,255,0,0.4); }} 
      .c-red   {{ color:#ff3333; text-shadow: 0 0 8px rgba(255,51,51,0.4); }}
      .c-yellow{{ color:#ffff00; text-shadow: 0 0 8px rgba(255,255,0,0.4); }}
      .c-white {{ color:#ffffff; text-shadow: 0 0 8px rgba(255,255,255,0.4); }}
      .blink {{ animation:blink 0.6s infinite; }} 
      @keyframes blink {{ 50% {{ opacity:0.2; }} }}
    </style></head><body>
    <div class="monitor">
      <div class="wave-area"><canvas id="monitorCanvas"></canvas></div>
      <div class="vitals">
        <div class="vital-item"><div class="vital-label c-green">HR</div><span id="hr" class="vital-value c-green">--</span><span class="vital-unit">bpm</span></div>
        <div class="vital-item"><div class="vital-label c-red">ABP</div><span id="bp" class="vital-value c-red">--/--</span><span class="vital-unit">mmHg</span></div>
        <div class="vital-item"><div class="vital-label c-yellow">SpO₂</div><span id="spo2" class="vital-value c-yellow">--</span><span class="vital-unit">%</span></div>
        <div class="vital-item"><div class="vital-label c-white">RESP</div><span id="rr" class="vital-value c-white">--</span><span class="vital-unit">/min</span></div>
      </div>
    </div>
    <script>
      const HR={hr}, SPO2={spo2}, RR={rr}, BP_SYS={bp_sys}, BP_DIA={bp_dia}, STATUS="{status}";
      const canvas = document.getElementById('monitorCanvas'); const ctx = canvas.getContext('2d'); let W, H;
      const resizeObserver = new ResizeObserver(entries => {{ for (let entry of entries) {{ W = entry.contentRect.width; H = entry.contentRect.height; if(W === 0 || H === 0) continue; canvas.width = W * devicePixelRatio; canvas.height = H * devicePixelRatio; canvas.style.width = W + 'px'; canvas.style.height = H + 'px'; ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0); }} }});
      resizeObserver.observe(canvas.parentElement);
      let beatCount = 0;
      function ecgSample(p) {{ let v = 0; v += 0.12 * Math.exp(-Math.pow((p - 0.1) / 0.02, 2)); let isPVC = false; if (STATUS === "critical") {{ if (beatCount % 5 === 0 && p < 0.4) {{ isPVC = true; }} }} if (isPVC) {{ v -= 0.5 * Math.exp(-Math.pow((p - 0.25) / 0.04, 2)); v += 1.5 * Math.exp(-Math.pow((p - 0.3) / 0.05, 2)); v -= 0.4 * Math.exp(-Math.pow((p - 0.4) / 0.06, 2)); }} else {{ v -= 0.15 * Math.exp(-Math.pow((p - 0.24) / 0.008, 2)); v += 1.0 * Math.exp(-Math.pow((p - 0.26) / 0.01, 2)); v -= 0.3 * Math.exp(-Math.pow((p - 0.28) / 0.012, 2)); let tAmp = 0.25; if (STATUS === "danger") tAmp = 0.1; if (STATUS === "critical") tAmp = -0.15; v += tAmp * Math.exp(-Math.pow((p - 0.45) / 0.06, 2)); }} return v; }}
      function abpSample(p) {{ let amp = 1.0; let notch = 0.15; if (STATUS === "danger") {{ amp = 0.6; notch = 0.05; }} if (STATUS === "critical") {{ amp = 0.2; notch = 0; }} let v; if (p < 0.12) {{ v = Math.sin((p / 0.12) * Math.PI / 2); }} else if (p < 0.45) {{ v = 1.0 - (p - 0.12) * 1.4 + notch * Math.exp(-Math.pow((p - 0.3) / 0.04, 2)); }} else {{ v = Math.max(0, 0.55 - (p - 0.45) * 0.8); }} return v * amp; }}
      function plethSample(p) {{ let amp = 1.0; let noise = 0; if (STATUS === "danger") {{ amp = 0.5; }} if (STATUS === "critical") {{ amp = 0.15; noise = (Math.random()-0.5)*0.1; }} let peak1 = Math.exp(-Math.pow((p - 0.15) / 0.1, 2)); let peak2 = 0.35 * Math.exp(-Math.pow((p - 0.4) / 0.12, 2)); return (peak1 + peak2) * amp + noise; }}
      function respSample(p) {{ let v = Math.sin(p * Math.PI * 2 - Math.PI / 2); if (STATUS === "critical") {{ v = v * 0.5 + 0.5 * Math.sin(p * Math.PI * 0.3) * Math.sin(p * Math.PI * 2); }} else if (STATUS === "danger") {{ v = v * 0.7; }} return v; }}
      const BUFFER = 600; const ecgData = new Array(BUFFER).fill(0), abpData = new Array(BUFFER).fill(0), plethData = new Array(BUFFER).fill(0), respData = new Array(BUFFER).fill(0);
      let ecgPhase = 0, abpPhase = 0, plethPhase = 0, respPhase = 0; const sampleInterval = 4; let lastTime = performance.now();
      function drawGrid(h4) {{ ctx.lineWidth = 0.5; ctx.strokeStyle = "rgba(255,255,255,0.03)"; for (let x=0; x<W; x+=15) {{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }} for (let i=0; i<4; i++) {{ let y = i * h4; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }} }}
      function drawWave(data, h_offset, h_region, color, amp) {{ const step = W / BUFFER; const midY = h_offset + h_region * 0.55; ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.shadowBlur = 5; ctx.shadowColor = color; for (let i=0; i<BUFFER; i++) {{ const x = i*step, y = midY - data[i]*amp; i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y); }} ctx.stroke(); ctx.shadowBlur = 0; }}
      function loop() {{ if (!W || !H) {{ requestAnimationFrame(loop); return; }} const now = performance.now(); let dt = now - lastTime; lastTime = now; if (dt > 100) dt = 100; let pointsToAdd = Math.floor(dt / sampleInterval); if (pointsToAdd > 50) pointsToAdd = 50; const cycleDurECG = 60 / HR, cycleDurResp = 60 / RR; const dPhaseECG = (sampleInterval / 1000) / cycleDurECG; const dPhaseResp = (sampleInterval / 1000) / cycleDurResp; for (let i = 0; i < pointsToAdd; i++) {{ let prevPhase = ecgPhase; ecgPhase = (ecgPhase + dPhaseECG) % 1; abpPhase = (abpPhase + dPhaseECG) % 1; plethPhase = (plethPhase + dPhaseECG) % 1; respPhase = (respPhase + dPhaseResp) % 1; if (ecgPhase < prevPhase) beatCount++; let v_ecg = ecgSample(ecgPhase); let v_abp = abpSample(abpPhase); let v_pleth = plethSample(plethPhase); let v_resp = respSample(respPhase); if (STATUS === "warn") {{ v_ecg += Math.sin(respPhase * Math.PI * 2) * 0.05; }} else if (STATUS === "danger") {{ v_ecg += Math.sin(respPhase * Math.PI * 2) * 0.1 + (Math.random()-0.5)*0.05; v_abp += (Math.random()-0.5)*0.02; }} else if (STATUS === "critical") {{ v_ecg += Math.sin(respPhase * Math.PI * 2) * 0.3 + (Math.random()-0.5)*0.1; v_abp += (Math.random()-0.5)*0.05; }} ecgData.push(v_ecg); ecgData.shift(); abpData.push(v_abp); abpData.shift(); plethData.push(v_pleth); plethData.shift(); respData.push(v_resp); respData.shift(); }} ctx.clearRect(0,0,W,H); const h4 = H / 4; drawGrid(h4); let ampScale = STATUS === "critical" ? 0.8 : 1.0; drawWave(ecgData, 0, h4, "#00ff00", h4*0.35*ampScale); drawWave(abpData, h4, h4, "#ff3333", h4*0.35*ampScale); drawWave(plethData, h4*2, h4, "#ffff00", h4*0.35*ampScale); drawWave(respData, h4*3, h4, "#ffffff", h4*0.35*ampScale); requestAnimationFrame(loop); }}
      function jit(base, amt) {{ return Math.round(base + (Math.random()-0.5)*amt); }}
      function updateVitals() {{ document.getElementById('hr').textContent = jit(HR, 2); document.getElementById('spo2').textContent = jit(SPO2, 1); document.getElementById('rr').textContent = jit(RR, 1); document.getElementById('bp').textContent = jit(BP_SYS, 2) + "/" + jit(BP_DIA, 1); if (STATUS==="critical"||STATUS==="danger") {{ document.getElementById('hr').classList.add('blink'); }} else {{ document.getElementById('hr').classList.remove('blink'); }} }}
      updateVitals(); setInterval(updateVitals, 1000); requestAnimationFrame(loop);
    </script></body></html>"""
    components.html(html_code, height=290, scrolling=False)

def _try_unlock(key: str):
    if key not in st.session_state.achievements: st.session_state.achievements.append(key)

def check_achievements(case, case_id: str):
    ss = st.session_state
    if getattr(ss, "game_over", False) and getattr(ss, "disease_progress", 100) < 100: ss.completed_cases.add(case_id)
    for aid, a in GLOBAL_ACHIEVEMENTS.items():
        try:
            if a["check"](ss, case): _try_unlock(f"global:{aid}")
        except Exception: pass
    for aid, a in CASE_ACHIEVEMENTS.get(case_id, {}).items():
        try:
            if a["check"](ss, case): _try_unlock(f"{case_id}:{aid}")
        except Exception: pass

def achievement_total() -> int: return len(flatten_achievements())
def achievement_unlocked_count() -> int: return len(st.session_state.get("achievements", []))

def render_achievement_grid(ach_dict: dict, prefix: str, cols_per_row: int = 2):
    unlocked = st.session_state.get("achievements", [])
    items = list(ach_dict.items())
    cols = st.columns(cols_per_row)
    for i, (aid, a) in enumerate(items):
        key = f"{prefix}:{aid}"; done = key in unlocked
        with cols[i % cols_per_row]:
            st.markdown(f"""<div style="background:{"rgba(14,165,233,0.12)" if done else "rgba(128,128,128,0.08)"}; border:1px solid {"#0ea5e9" if done else "rgba(150,150,150,0.35)"}; border-radius:14px; padding:14px; margin-bottom:10px; display:flex; align-items:center; gap:12px;"><div style="font-size:34px; {"" if done else "filter:grayscale(1) opacity(0.35);"}">{a['icon']}</div><div style="flex:1;"><div style="font-weight:800; font-size:15px; color:{"#0ea5e9" if done else "#8a94a6"};">{a['name']}</div><div style="font-size:12px; color:{"#cbd5e1" if done else "#6b7280"}; margin-top:2px;">{html.escape(a['desc'])}</div><div style="font-size:11px; color:{"#0ea5e9" if done else "#8a94a6"}; margin-top:4px;">{"✅ 已解锁" if done else "🔒 未解锁"}</div></div></div>""", unsafe_allow_html=True)

def render_sidebar_achievements(case_id: str):
    st.markdown("### 🏅 成就进度")
    total = achievement_total(); got = achievement_unlocked_count()
    st.progress(min(1.0, got / total) if total else 0)
    st.caption(f"已解锁 **{got} / {total}**")
    if case_id in CASE_ACHIEVEMENTS:
        st.caption("本病例成就：")
        cols = st.columns(4); unlocked = st.session_state.get("achievements", [])
        for i, (aid, a) in enumerate(CASE_ACHIEVEMENTS[case_id].items()):
            done = f"{case_id}:{aid}" in unlocked
            cols[i].markdown(f'<div title="{html.escape(a["name"])}：{html.escape(a["desc"])}" style="text-align:center; font-size:24px; {"" if done else "filter:grayscale(1) opacity(0.35);"}">{a["icon"]}</div>', unsafe_allow_html=True)

def build_parent_prompt(case, trust_score, disease_progress, unlocked_clues, is_furious):
    facts = "\n".join(f"  · {c}" for c in case["clues"])
    said = "、".join(unlocked_clues) if unlocked_clues else "（还没聊到具体细节）"
    if is_furious: mood = "你现在极度愤怒、焦虑、绝望，觉得这医生不靠谱，想投诉换人。"
    elif trust_score >= 60: mood = "你已经稍微信任这个医生了，但还非常担心孩子。"
    elif trust_score <= 20: mood = "你心急如焚、带着哭腔，情绪濒临崩溃。"
    else: mood = "你非常焦急，但还勉强能配合医生问话。"
    if disease_progress >= 80: mood += " 你注意到孩子脸色越来越紫，你已经开始歇斯底里，声音发抖，语无伦次。"
    elif disease_progress >= 60: mood += " 你越来越慌，频繁回头看孩子，语速变快。"
    return f"""你正在扮演一位儿科急诊患儿的家长，名字叫「{case['patient']}的妈妈」。
{mood}
【你孩子的真实情况 —— 这是你必须遵守的事实，不许编造】
患儿：{case['patient']}
关键症状（你要用家长大白话说出来，不许用专业术语）：
{facts}
【已经告诉过医生的事】：{said}
【铁律】
1. 你是不懂医学的普通家长。绝对禁止说"三凹征""吸气性喉鸣""发绀"等任何专业词，必须用大白话。
2. 医生问什么，你就用「真实情况」里的事实回答什么。医生问到的细节如果事实清单里没有，就回答"没有"或"我不太清楚"。
3. 不要主动报病名，也不要说"医生您觉得是什么病"，你是来求助的。
4. 不要重复你上一句话的开头。每轮都要带一点新信息或新情绪。
5. 如果医生问的东西你刚说过，可以不耐烦地说"我刚才不是说了吗"。
6. 每次回复不超过 2 句话，像真人说话一样自然，可以带叹气、哭腔、反问。
7. 如果医生态度冷漠或凶你，你要发火；如果医生安慰你，你会稍微缓和。
"""

# ================= 7. 页面路由 =================
if st.session_state.screen == "start":
    st.markdown('<div style="text-align: center; font-size: 72px; font-weight: 900; color: #0ea5e9; margin-top: 80px; letter-spacing: 5px;">🏥 PediMind</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 38px; font-weight: 700; color: #ef4444; margin-bottom: 30px; letter-spacing: 5px;">急诊风云</div>', unsafe_allow_html=True)
    st.markdown("""<div style="text-align: center; max-width: 600px; margin: 0 auto 50px auto; line-height: 2; font-size: 16px;"><span style="color: #0ea5e9; font-weight: bold;">欢迎来到 PediMind 儿科急诊模拟器。</span><br>在这里，你是一名急诊科医生。<br>面对各种凶险的儿科急症，你的每一次决策，都将决定患儿的生死。</div>""", unsafe_allow_html=True)
    _, col_mid, _ = st.columns([1, 1, 1])
    with col_mid:
        st.text_input("👨‍⚕️ 你的昵称", key="player_name", max_chars=12)
        st.caption(f"专属 ID：`{st.session_state.player_id}`")
        if st.button("🚀 开始游戏", use_container_width=True, type="primary"):
            st.session_state.screen = "case_select"; st.rerun()
    st.stop()

if st.session_state.screen == "case_select":
    st.markdown('<div style="text-align: center; font-size: 42px; font-weight: 900; color: #0ea5e9; margin-top: 30px; margin-bottom: 30px;">📋 选择病例</div>', unsafe_allow_html=True)
    st.markdown("""<div style="padding: 20px; border-radius: 16px; background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%); color: white; margin-bottom: 30px; text-align: center; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);"><h3>✨ 挑战未知急诊（AI 实时生成）</h3><p style="font-size: 14px; opacity: 0.9;">打破死题库限制，让大模型为你随机生成一个全新的儿科急症！</p></div>""", unsafe_allow_html=True)
    if st.button("🎲 让 AI 生成随机病例", use_container_width=True, type="primary"):
        with st.spinner("🧠 AI 主任正在编写新病历..."):
            try:
                gen_prompt = """
                你是一个儿科急诊专家。请随机生成一个儿科急症病例。必须返回纯 JSON 格式，不要任何额外解释。
                包含字段："name", "disease", "patient", "description", "opening_line", "clues" (list of 4), "ai_role", "diagnosis" (dict A-D), "crisis" (dict A-C, each with label, type, disease, reply)。
                """
                res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": gen_prompt}], temperature=0.7)
                json_str = extract_json(res.choices[0].message.content)
                new_case = json.loads(json_str)
                st.session_state.current_case = "ai_generated"
                st.session_state.ai_case_data = new_case
                st.session_state.screen = "difficulty_select"
                st.rerun()
            except Exception as e:
                st.error(f"AI 生成失败：{e}")

    st.divider()
    st.markdown('<div style="font-size: 24px; font-weight: bold; margin-bottom: 15px;">📁 经典病例库</div>', unsafe_allow_html=True)
    for case_id, case in CASES_CONFIG.items():
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f'<div class="case-card"><div style="font-size: 24px; font-weight: bold; color: #0ea5e9; margin-bottom: 10px;">📁 {case["name"]}</div><div style="font-size: 14px; margin-bottom: 5px;"><b>患儿：</b>{case["patient"]}</div><div style="font-size: 14px; color: #ef4444; margin-bottom: 5px;"><b>难度：</b>{case["difficulty"]}</div><div style="font-size: 13px; font-style: italic; margin-top: 10px;">{case["description"]}</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("▶️ 进入病例", key=f"select_{case_id}", use_container_width=True, type="primary"):
                st.session_state.current_case = case_id; st.session_state.screen = "difficulty_select"; st.rerun()
    st.stop()

if st.session_state.screen == "difficulty_select":
    st.markdown('<div style="text-align: center; font-size: 42px; font-weight: 900; color: #0ea5e9; margin-top: 30px;">⚙️ 选择难度</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.info("🌱 **简单模式**\n- 6个行动点\n- 初始信任值40")
        if st.button("开始简单模式", use_container_width=True):
            st.session_state.game_mode = "easy"; st.session_state.screen = "game"; st.rerun()
    with col2:
        st.warning("🔥 **困难模式**\n- 4个行动点\n- 初始信任值20")
        if st.button("开始困难模式", use_container_width=True):
            st.session_state.game_mode = "hard"; st.session_state.screen = "game"; st.rerun()
    st.stop()

if st.session_state.screen == "achievements":
    st.markdown('<div style="text-align:center; font-size:42px; font-weight:900; color:#0ea5e9; margin:20px 0;">🏅 成就殿堂</div>', unsafe_allow_html=True)
    total = achievement_total(); got = achievement_unlocked_count()
    c1, c2, c3 = st.columns(3); c1.metric("成就总数", total); c2.metric("已解锁", got); c3.metric("完成度", f"{(got/total*100 if total else 0):.0f}%")
    st.progress(min(1.0, got / total) if total else 0); st.divider()
    tab_names = ["🌐 通用成就"] + [f"📁 {CASES_CONFIG[cid]['name']}" for cid in CASES_CONFIG]
    tabs = st.tabs(tab_names)
    with tabs[0]: render_achievement_grid(GLOBAL_ACHIEVEMENTS, "global")
    for i, cid in enumerate(CASES_CONFIG, start=1):
        with tabs[i]: render_achievement_grid(CASE_ACHIEVEMENTS.get(cid, {}), cid)
    st.stop()

if st.session_state.screen == "profile":
    st.markdown('<div style="text-align:center; font-size:42px; font-weight:900; color:#0ea5e9; margin:20px 0;">📕 我的错题本</div>', unsafe_allow_html=True)
    records = fetch_my_records(st.session_state.player_id)
    if not records: st.info("还没有记录。")
    else:
        df = pd.DataFrame(records)
        c1, c2, c3 = st.columns(3); c1.metric("总场次", len(df)); c2.metric("平均分", f"{df['final_score'].mean():.1f}"); c3.metric("存活率", f"{df['survived'].mean()*100:.0f}%")
        for r in records:
            mistakes = json.loads(r["mistakes"] or "[]")
            with st.expander(f"{'✅' if r['survived'] else '💀'} {r['created_at']} | {r['disease']} | 得分 {r['final_score']}"):
                st.write(f"**难度**：{r['mode']} | **病情度**：{max(0, r['disease_progress'])}")
                if mistakes: st.warning("**失误记录：**\n" + "\n".join(f"- {m}" for m in mistakes))
    st.stop()

if st.session_state.screen == "stats":
    st.markdown('<div style="text-align:center; font-size:42px; font-weight:900; color:#0ea5e9; margin:20px 0;">📊 全服游玩统计</div>', unsafe_allow_html=True)
    all_recs = fetch_all_records()
    if not all_recs: st.info("暂无任何玩家数据。")
    else:
        df = pd.DataFrame(all_recs)
        c1, c2, c3 = st.columns(3); c1.metric("总游玩场次", len(df)); c2.metric("独立玩家数", df["player_id"].nunique()); c3.metric("全服平均分", f"{df['final_score'].mean():.1f}")
        agg = df.groupby("disease").agg(场次=("id", "count"), 平均分=("final_score", "mean"), 存活率=("survived", "mean")).reset_index()
        agg["平均分"] = agg["平均分"].round(1); agg["存活率"] = (agg["存活率"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(agg, use_container_width=True, hide_index=True)
    st.stop()

# ================= 8. 游戏主逻辑 =================
if st.session_state.screen == "game":
    if st.session_state.current_case == "ai_generated":
        case = st.session_state.ai_case_data
        case.setdefault("clues", ["线索1", "线索2", "线索3", "线索4"])
        case.setdefault("ai_role", "焦急的家长")
        case.setdefault("diagnosis", {"A": case["disease"], "B": "其他疾病1", "C": "其他疾病2", "D": "其他疾病3"})
        case.setdefault("crisis", {"A": {"label": "正确操作", "type": "correct", "disease": -30, "reply": "好转"}, "B": {"label": "错误操作", "type": "harmful", "disease": 30, "reply": "恶化"}})
        case.setdefault("audio", "wheezing.mp3")
        case.setdefault("avatar", "static/mom.png")
    else:
        case = CASES_CONFIG[st.session_state.current_case]
    
    mode = st.session_state.game_mode

    defaults = {
        "messages": [{"role": "assistant", "content": case.get("opening_line", "医生！快看看孩子！")}],
        "chat_history": [{"role": "assistant", "content": case.get("opening_line", "医生！快看看孩子！")}],
        "trust_score": 40 if mode == "easy" else 20,
        "action_points": 6 if mode == "easy" else 4, 
        "disease_progress": 30, "game_over": False, "time_period": 1, "unlocked_clues": [],
        "penalty_log": [], "crisis_correct_count": 0, "final_score": 0, "final_title": "",
        "score_inquiry": 0, "score_diagnosis": 0, "score_emergency": 0, "score_empathy": 0,
        "diagnosis_made": None, "diagnosis_processed": False,
        "auscultation_mode": False, "auscultation_completed": False,
        "visual_exam_mode": False, 
        "physical_exam_done": {}, "comm_made": False,
        "call_senior_used": False, "senior_mode": False,
        "crisis_start_time": time.time(), "record_saved": False,
        "ai_review": None, "ai_review_generated": False,
        "calming_down": False,
        "lab_exams_done": [], 
        "my_hypotheses": [] 
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    check_achievements(case, st.session_state.current_case)
    dp = st.session_state.get("disease_progress", 0)
    
    if dp >= 90 and not st.session_state.get("game_over", False) and st.session_state.time_period != 3:
        alert_css = """<style>@keyframes full-red-pulse {0% {background-color: rgba(255,0,0,0.1);}50% {background-color: rgba(255,0,0,0.4);}100% {background-color: rgba(255,0,0,0.1);}} #critical-overlay {position: fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:999999;animation: full-red-pulse 0.8s infinite;}</style><div id="critical-overlay"></div>"""
        st.markdown(alert_css, unsafe_allow_html=True)
    elif dp >= 70 and not st.session_state.get("game_over", False) and st.session_state.time_period != 3:
        alert_css = """<style>@keyframes full-warn-pulse {0% {background-color: rgba(255,100,0,0.05);}50% {background-color: rgba(255,100,0,0.2);}100% {background-color: rgba(255,100,0,0.05);}} #warn-overlay {position: fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:999999;animation: full-warn-pulse 1.5s infinite;}</style><div id="warn-overlay"></div>"""
        st.markdown(alert_css, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### 🩺 病例：{case['name']}")
        render_sidebar_achievements(st.session_state.current_case)

    # 顶部极简仪表盘
    st.markdown(f"""
    <div style="display: flex; gap: 20px; align-items: center; background: {"rgba(255,255,255,0.03)" if st.session_state.theme == "night" else "#ffffff"}; padding: 15px 25px; border-radius: 16px; border: 1px solid {"rgba(255,255,255,0.08)" if st.session_state.theme == "night" else "#e2e8f0"}; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);">
        <div style="flex: 1; display: flex; justify-content: space-between; align-items: center;">
            <div><span style="font-size: 13px; color: #64748b; font-weight: 600;">❤️ 信任值</span> <span style="font-size: 22px; font-weight: 800; color: {"#ef4444" if st.session_state.trust_score <= 15 else "#0ea5e9"};">{st.session_state.trust_score}</span></div>
            <div><span style="font-size: 13px; color: #64748b; font-weight: 600;">⚡ 体力点</span> <span style="font-size: 22px; font-weight: 800; color: #eab308;">{st.session_state.action_points}</span></div>
            <div><span style="font-size: 13px; color: #64748b; font-weight: 600;">⚠️ 病情度</span> <span style="font-size: 22px; font-weight: 800; color: #ef4444;">{max(0, st.session_state.disease_progress)}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2.8, 1])

    # ================= 左侧面板 =================
    with col_left:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        render_monitor(dp)
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        
        if dp < 40: st.caption("🟢 意识：轻度烦躁")
        elif dp < 70: st.caption("🟡 意识：明显烦躁")
        elif dp < 90: st.caption("🟠 意识：发绀、烦躁")
        else: st.caption("🔴 意识：意识模糊")
        
        st.divider()
        
        if not st.session_state.game_over and st.session_state.time_period == 1:
            st.caption(f"💡 剩余体力：{st.session_state.action_points}")
            
            if st.button("📝 结束接诊，给出诊断", use_container_width=True, type="primary"):
                st.session_state.time_period = 2
                st.rerun()
                
            if st.session_state.disease_progress >= 80:
                st.error("🚨 极度危险！")
                if st.button("🚨 患儿突发恶化，强制抢救", use_container_width=True):
                    st.session_state.time_period = 3
                    st.session_state.action_points = 3 if st.session_state.game_mode == "easy" else 2
                    st.session_state.crisis_start_time = time.time()
                    st.session_state.messages.append({"role": "assistant", "content": "（患儿突然剧烈哭闹，呼吸极度困难！）医生救命啊！"})
                    st.rerun()
            elif st.session_state.action_points <= 0:
                st.warning("⚠️ 体力耗尽，时间拖得太久了！")
                if st.button("🚨 紧急抢救", use_container_width=True):
                    st.session_state.time_period = 3
                    st.session_state.action_points = 3 if st.session_state.game_mode == "easy" else 2
                    st.session_state.crisis_start_time = time.time()
                    st.session_state.disease_progress = max(75, st.session_state.disease_progress + 10)
                    st.session_state.messages.append({"role": "assistant", "content": "（时间被拖得太久，患儿脸色发紫！）医生，快救救他！"})
                    st.rerun()

        st.divider()
        
        if not st.session_state.game_over and not st.session_state.call_senior_used and st.session_state.time_period == 1:
            if st.button("🆘 呼叫上级医生 (消耗1体力)", use_container_width=True):
                if st.session_state.action_points <= 0: st.error("你已经没有体力去打电话了！")
                else:
                    st.session_state.action_points -= 1; st.session_state.trust_score -= 10
                    st.session_state.call_senior_used = True; st.session_state.senior_mode = True; st.rerun()
        elif st.session_state.call_senior_used: st.caption("☎️ 上级医生已下班...")

        if st.button("🔄 重新开始", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "screen", "current_case", "game_mode", "player_id", "player_name", "achievements", "completed_cases"]: del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()

    # ================= 右侧面板 =================
    with col_right:
        tab1, tab2 = st.tabs(["🔍 线索", "💭 鉴别"])
        
        with tab1:
            if mode == "hard":
                for i, clue in enumerate(case["clues"]):
                    if clue in st.session_state.unlocked_clues: st.success(f"✅ {clue}")
                    else: st.text(f"🔒 未知线索 {i+1}/{len(case['clues'])}")
            else:
                for clue in case["clues"]:
                    if clue in st.session_state.unlocked_clues: st.success(f"✅ {clue}")
                    else: st.text("🔒 未知线索")
            if st.session_state.penalty_log:
                st.divider(); st.caption("📝 失误")
                for log in st.session_state.penalty_log[-2:]: st.warning(log)
                
        with tab2:
            st.caption("记录你的临床思路：")
            hyp_options = list(case["diagnosis"].values())
            st.session_state.my_hypotheses = st.multiselect("怀疑疾病", hyp_options, default=st.session_state.my_hypotheses, label_visibility="collapsed")
            if len(st.session_state.my_hypotheses) == 0: st.caption("💡 *建议勾选 1-2 个怀疑的疾病。*")
            elif len(st.session_state.my_hypotheses) > 2: st.warning("⚠️ 怀疑太多了，请聚焦！")
            else: st.info("💡 正在验证上述怀疑...")
            if case.get("disease") in st.session_state.my_hypotheses:
                if "三凹征" in st.session_state.unlocked_clues or "流涎" in st.session_state.unlocked_clues:
                    st.success("✅ 线索吻合，高度怀疑！")

    # ================= 中间核心交互区 =================
    with col_center:
        if not st.session_state.game_over:
            if st.session_state.time_period == 1: 
                st.markdown("<h2 style='margin-bottom: 0;'>🎬 第一幕：急诊初探</h2>", unsafe_allow_html=True)
                st.caption("任务：自由问诊与查体，收集线索。体力耗尽或病情恶化将强制进入抢救！")
            elif st.session_state.time_period == 2: 
                st.markdown("<h2 style='margin-bottom: 0;'>🎬 第二幕：一锤定音</h2>", unsafe_allow_html=True)
            elif st.session_state.time_period == 3: 
                st.markdown("<h2 style='margin-bottom: 0;'>🎬 第三幕：生死时速</h2>", unsafe_allow_html=True)
        else: 
            st.markdown("<h2 style='margin-bottom: 0;'>🎬 第四幕：结案复盘</h2>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)

        # 家属崩溃检测
        if st.session_state.get("trust_score", 0) >= 20: st.session_state.calming_down = False
        if st.session_state.get("trust_score", 0) <= 10 and not st.session_state.get("calming_down", False) and not st.session_state.game_over and st.session_state.time_period == 1:
            st.error("🚨 **极度危险：家属情绪彻底崩溃，拒绝配合！**")
            st.markdown("**家属哭喊咆哮**：“你到底会不会看病！刚才还好好的，怎么越看越重了！我们要换医生！！”")
            col_calm1, col_calm2, col_calm3 = st.columns(3)
            with col_calm1:
                if st.button("1. （共情安抚）\n我理解您，孩子生病您最难受，请相信我。", use_container_width=True):
                    st.session_state.trust_score += 20; st.session_state.calming_down = True
                    st.session_state.disease_progress += 5; st.toast("✅ 家属情绪勉强平复...", icon="💬"); st.rerun()
            with col_calm2:
                if st.button("2. （强词夺理）\n病情本来就会恶化，你得听我的！", use_container_width=True):
                    st.session_state.trust_score -= 15; st.session_state.disease_progress += 15
                    st.session_state.penalty_log.append("沟通失败引发争吵")
                    if check_medical_dispute(): st.rerun()
                    st.rerun()
            with col_calm3:
                if st.button("3. （冷漠硬刚）\n别吵了，不治就赶紧办出院！", use_container_width=True):
                    st.session_state.trust_score = 0; check_medical_dispute(); st.rerun()
            st.stop()

        # 查体工具箱
        if not st.session_state.game_over and st.session_state.time_period == 1:
            with st.expander("🩺 体格检查与辅助检查工具箱（消耗体力）", expanded=False):
                tab1, tab2, tab3, tab4 = st.tabs(["👁️ 视诊", "👂 听诊", "🖐️ 触诊", "🧪 辅助检查"])
                with tab1:
                    if st.button("开始视诊 (消耗1体力)", use_container_width=True):
                        if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("vis"):
                            st.session_state.action_points -= 1; st.session_state.physical_exam_done["vis"] = True
                            st.session_state.visual_exam_mode = True; st.rerun()
                        elif st.session_state.action_points <= 0: st.error("体力不足！")
                with tab2:
                    if st.button("进行听诊 (消耗1体力)", use_container_width=True):
                        if st.session_state.action_points > 0 and not st.session_state.auscultation_completed:
                            st.session_state.action_points -= 1; st.session_state.auscultation_mode = True; st.rerun()
                        elif st.session_state.action_points <= 0: st.error("体力不足！")
                with tab3:
                    if st.button("进行触诊 (消耗1体力)", use_container_width=True):
                        if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("palp"):
                            st.session_state.action_points -= 1; st.session_state.physical_exam_done["palp"] = True
                            if st.session_state.current_case in ["laryngitis", "epiglottitis"]:
                                st.session_state.disease_progress = min(100, st.session_state.disease_progress + 15)
                                st.session_state.trust_score -= 15
                                st.session_state.penalty_log.append("不必要的触诊激惹患儿")
                                st.session_state.messages.append({"role": "assistant", "content": "（警示：患儿因你的触碰极度恐惧，剧烈哭闹挣扎，家长一把推开你：“你会不会看病！”）"})
                                if check_medical_dispute(): st.rerun()
                            elif st.session_state.current_case == "foreign_body":
                                st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5); st.session_state.unlocked_clues.append("呼吸音不对称")
                                st.session_state.messages.append({"role": "assistant", "content": "（触诊：气管居中，语颤减弱。叩诊：右侧呈过清音，提示异物阻塞。）"})
                            st.rerun()
                        elif st.session_state.action_points <= 0: st.error("体力不足！")
                with tab4:
                    if st.button("开具 X线/影像学检查 (消耗1体力)", use_container_width=True):
                        if st.session_state.action_points > 0 and "xray" not in st.session_state.lab_exams_done:
                            st.session_state.action_points -= 1; st.session_state.lab_exams_done.append("xray")
                            if st.session_state.current_case == "epiglottitis":
                                st.session_state.disease_progress += 40; st.session_state.trust_score -= 20
                                st.session_state.penalty_log.append("会厌炎强行拍片导致气道梗阻")
                                st.session_state.messages.append({"role": "assistant", "content": "（在放射科摆体位时，患儿突然剧烈哭闹，呼吸骤停！绝对不能在怀疑会厌炎时强行搬动患儿！）"})
                                if check_medical_dispute(): st.rerun()
                            elif st.session_state.current_case == "foreign_body":
                                st.session_state.messages.append({"role": "assistant", "content": "（X线回报：右主支气管可见高密度异物影。）"})
                                if "呼吸音不对称" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("呼吸音不对称")
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": "（检查回报：影像学未见明显异常。）"})
                            st.rerun()
                        elif st.session_state.action_points <= 0: st.error("体力不足！")

        # 上级医生提问
        if st.session_state.get("senior_mode", False):
            st.info("📞 电话那头传来了上级医生疲惫但沉稳的声音：")
            q, options, correct_idx = "", [], 0
            if st.session_state.current_case == "laryngitis":
                q = "患儿呼吸困难，如果是急性喉炎，你觉得查体最关键的一步是什么？"
                options = ["A. 赶紧用压舌板看嗓子", "B. 观察有没有三凹征和听吸气性喉鸣", "C. 做全身触诊"]; correct_idx = 1
            elif st.session_state.current_case == "foreign_body":
                q = "孩子憋成这样，既然怀疑异物，首选什么动作？"
                options = ["A. 立即海姆立克急救", "B. 去放射科拍片", "C. 打镇静剂"]; correct_idx = 0
            elif st.session_state.current_case == "asthma":
                q = "哮喘急性发作，最立竿见影的给药途径是什么？"
                options = ["A. 静脉推注抗生素", "B. 口服止咳糖浆", "C. 雾化吸入沙丁胺醇"]; correct_idx = 2
            elif st.session_state.current_case == "epiglottitis":
                q = "高度怀疑会厌炎，孩子一直流口水，能让他平躺检查吗？"
                options = ["A. 可以，平躺更好查", "B. 绝对不行，必须保持坐位", "C. 绑起来强行查"]; correct_idx = 1
            else:
                q = "面对不明原因的急症，最重要的第一步是什么？"
                options = ["A. 盲目给药", "B. 维持气道和生命体征稳定", "C. 做全套检查"]; correct_idx = 1
                
            st.write(f"**上级提问：{q}**")
            for i, opt in enumerate(options):
                if st.button(opt, key=f"senior_{i}", use_container_width=True):
                    if i == correct_idx:
                        st.success("📞 上级医生：“对，算你脑子清醒！赶紧处理，我马上到！”")
                        st.session_state.disease_progress = max(0, st.session_state.disease_progress - 20)
                        if st.session_state.current_case in ["laryngitis", "epiglottitis"]:
                            if "三凹征" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("三凹征")
                        elif st.session_state.current_case == "foreign_body":
                            if "进食呛咳史" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("进食呛咳史")
                        elif st.session_state.current_case == "asthma":
                            if "双肺哮鸣音" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("双肺哮鸣音")
                        st.session_state.score_inquiry += 10
                    else:
                        st.error("📞 上级医生：“你平时书都读哪去了？赶紧自己想办法！”（电话挂断）")
                        st.session_state.disease_progress = min(100, st.session_state.disease_progress + 10)
                        st.session_state.trust_score -= 10
                        st.session_state.penalty_log.append("上级指导时答错")
                    st.session_state.senior_mode = False; st.rerun()
            st.stop()

        # ================= 视诊 =================
        if st.session_state.get("visual_exam_mode", False):
            st.warning("👁️ 视诊中：请仔细观察，并在下方输入您的发现。")
            if st.session_state.current_case == "laryngitis":
                if os.path.exists("static/three_depressions.mp4"): st.video("static/three_depressions.mp4")
                else: st.info("🎥 [视频占位：患儿呼吸极度费力，胸口和脖子下方随着呼吸深深凹陷]")
            elif st.session_state.current_case == "epiglottitis": st.info("🎥 [场景：患儿呈端坐位，极度惊恐，口水顺着嘴角不断流下，完全无法躺下]")
            else: st.info("🎥 [场景：患儿端坐呼吸，满头大汗，嘴唇发紫，鼻翼扇动]")

            st.write("### 📝 请填写您的视诊报告：")
            exam_finding = st.text_input("1. 您观察到了什么异常体征？（如：三凹征、流涎、发绀等）", key="vis_input_1")
            exam_diagnosis = st.text_input("2. 结合体征，您的初步病情判断是什么？（如：上气道梗阻、异物等）", key="vis_input_2")

            if st.button("提交视诊报告", type="primary", use_container_width=True):
                if not exam_finding.strip(): st.error("请至少填写您观察到的异常体征！")
                else:
                    finding_str = exam_finding.lower().replace(" ", ""); diag_str = exam_diagnosis.lower().replace(" ", "")
                    is_correct = False
                    if st.session_state.current_case == "laryngitis":
                        if any(k in finding_str for k in ["三凹", "凹陷", "锁骨", "胸骨", "吸气"]) or any(k in diag_str for k in ["梗阻", "喉炎"]): is_correct = True
                    elif st.session_state.current_case == "epiglottitis":
                        if any(k in finding_str for k in ["流涎", "口水", "端坐", "拒绝平卧"]) or any(k in diag_str for k in ["会厌", "梗阻"]): is_correct = True
                    elif st.session_state.current_case in ["foreign_body", "asthma"]:
                        if any(k in finding_str for k in ["发绀", "紫", "三凹", "呼气", "端坐"]) or any(k in diag_str for k in ["异物", "哮喘", "梗阻"]): is_correct = True
                    
                    if is_correct:
                        st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                        st.session_state.score_inquiry += 10
                        st.session_state.messages.append({"role": "assistant", "content": f"（视诊记录：{exam_finding}。你敏锐地察觉到了病情的关键！）"})
                        if st.session_state.current_case == "laryngitis" and "三凹征" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("三凹征")
                        elif st.session_state.current_case == "epiglottitis":
                            if "流涎" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("流涎")
                            if "拒绝平卧" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("拒绝平卧")
                        elif st.session_state.current_case in ["foreign_body", "asthma"]:
                            if "面色发绀" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("面色发绀")
                        st.toast("✅ 视诊准确！你抓住了核心体征。", icon="🩺")
                    else:
                        st.session_state.trust_score -= 5
                        st.session_state.penalty_log.append(f"视诊未抓住重点：{exam_finding}")
                        st.session_state.messages.append({"role": "assistant", "content": "（家属看你观察了半天，焦急地问：“大夫，到底看出啥问题没啊？”）"})
                        st.toast("❌ 观察不够准确。", icon="⚠️")
                    st.session_state.visual_exam_mode = False; st.rerun()

        # ================= 听诊 =================
        elif st.session_state.auscultation_mode:
            st.warning("🩺 听诊音频：")
            audio_path = f"static/{case['audio']}"
            if os.path.exists(audio_path): st.audio(audio_path, format="audio/mp3")
            else:
                st.info("📢 检测到本地音频缺失，已连接【虚拟听诊器】。")
                audio_html = """<div style="background:#1e1e1e; padding:15px; border-radius:10px; text-align:center; color:white; border: 2px solid #333;"><p style="color:#00ff00; font-family:monospace; font-size:16px; margin-bottom:15px;">🩺 虚拟听诊器已连接 (点击播放)</p><button onclick="playStridor()" style="background:#4CAF50; color:white; padding:10px 20px; border:none; border-radius:8px; cursor:pointer; margin-right:15px; font-weight:bold;">▶ 播放 吸气性喉鸣</button><button onclick="playWheezing()" style="background:#FFC107; color:#333; padding:10px 20px; border:none; border-radius:8px; cursor:pointer; font-weight:bold;">▶ 播放 呼气性哮鸣音</button></div><script>let audioCtx = null;function initAudio() { if (!audioCtx) { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } if (audioCtx.state === 'suspended') { audioCtx.resume(); } }function playStridor() { initAudio(); const osc = audioCtx.createOscillator(); const gain = audioCtx.createGain(); osc.type = 'sawtooth'; osc.frequency.setValueAtTime(1200, audioCtx.currentTime); const lfo = audioCtx.createOscillator(); lfo.type = 'sine'; lfo.frequency.value = 8; const lfoGain = audioCtx.createGain(); lfoGain.gain.value = 300; lfo.connect(lfoGain); lfoGain.connect(osc.frequency); gain.gain.setValueAtTime(0, audioCtx.currentTime); gain.gain.linearRampToValueAtTime(0.08, audioCtx.currentTime + 0.1); gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 1.5); osc.connect(gain); gain.connect(audioCtx.destination); lfo.start(); osc.start(); osc.stop(audioCtx.currentTime + 1.5); }function playWheezing() { initAudio(); const osc = audioCtx.createOscillator(); const gain = audioCtx.createGain(); osc.type = 'triangle'; osc.frequency.setValueAtTime(400, audioCtx.currentTime); const lfo = audioCtx.createOscillator(); lfo.type = 'sine'; lfo.frequency.value = 4; const lfoGain = audioCtx.createGain(); lfoGain.gain.value = 80; lfo.connect(lfoGain); lfoGain.connect(osc.frequency); gain.gain.setValueAtTime(0, audioCtx.currentTime); gain.gain.linearRampToValueAtTime(0.15, audioCtx.currentTime + 0.2); gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 2.0); osc.connect(gain); gain.connect(audioCtx.destination); lfo.start(); osc.start(); osc.stop(audioCtx.currentTime + 2.0); }</script>"""
                components.html(audio_html, height=120)
            st.write("### 请判断这是什么声音：")
            if st.button("吸气性喉鸣 (Stridor)", use_container_width=True):
                if case["disease"] in ("急性感染性喉炎", "急性会厌炎"):
                    st.session_state.unlocked_clues.append("吸气性喉鸣"); st.session_state.score_diagnosis += 10; st.toast("✅ 判断正确！", icon="🎯")
                else: st.session_state.trust_score -= 5; st.toast("❌ 判断错误。", icon="❌")
                st.session_state.auscultation_mode = False; st.session_state.auscultation_completed = True; st.rerun()
            if st.button("呼气性哮鸣音 (Wheezing)", use_container_width=True):
                if case["disease"] == "支气管哮喘急性发作":
                    st.session_state.unlocked_clues.append("双肺哮鸣音"); st.session_state.score_diagnosis += 10; st.toast("✅ 判断正确！", icon="🎯")
                else: st.session_state.trust_score -= 5; st.toast("❌ 判断错误。", icon="❌")
                st.session_state.auscultation_mode = False; st.session_state.auscultation_completed = True; st.rerun()

        # ================= 聊天与初步诊断 =================
        else:
            @st.fragment
            def chat_interface():
                with st.container(border=True):
                    with st.container(height=380):
                        for msg in st.session_state.messages: render_message(msg["role"], msg["content"], case)

                    if not st.session_state.game_over:
                        if st.session_state.time_period == 1:
                            st.markdown('<p style="font-size:12px; color:#94a3b8; margin:10px 0 5px 0;">快速提问</p>', unsafe_allow_html=True)
                            q_cols = st.columns(3, gap="small")
                            if st.session_state.current_case == "laryngitis": quick_prompts = ["咳嗽声音什么样？", "晚上更严重吗？", "少废话，赶紧办住院！"]
                            else: quick_prompts = ["吃了什么东西？", "有没有脸色发紫？", "别哭了，到底怎么回事！"]
                            
                            prompt = None
                            for i, q in enumerate(quick_prompts):
                                if q_cols[i].button(q, use_container_width=True, key=f"quick_{i}"): prompt = q

                            with st.form(key="chat_form", clear_on_submit=True):
                                col_input, col_btn = st.columns([5.5, 1])
                                with col_input: chat_val = st.text_input("输入", placeholder="输入问诊、查体要求或安抚话语...", label_visibility="collapsed")
                                with col_btn: submitted = st.form_submit_button("发送 ➤", use_container_width=True)
                            if submitted and chat_val: prompt = chat_val

                            if prompt:
                                if st.session_state.action_points <= 0: st.warning("体力已用完，请尽快在左侧点击【结束接诊】。")
                                else:
                                    st.session_state.action_points -= 1
                                    st.session_state.messages.append({"role": "user", "content": prompt})
                                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                                    
                                    is_furious = st.session_state.trust_score <= 15
                                    cold_words = ["闭嘴", "烦死了", "别吵", "赶紧的", "少废话", "按我说的做", "滚", "出去", "别哭了"]
                                    empathy_words = ["别急", "不要急", "理解", "放心", "辛苦", "我知道", "我懂", "相信", "安抚", "冷静"]
                                    medical_keywords = ["疼", "痛", "咳", "喘", "烧", "吃", "喝", "拉", "吐", "睡", "精神", "怎么", "多久", "什么", "哪", "时", "检查", "听", "看", "摸", "病", "症", "大小便", "颜色", "呼吸", "胸口", "晚上", "半夜", "夜里", "加重", "白天", "过敏", "呛"]
                                    
                                    is_empathy = any(w in prompt for w in empathy_words)
                                    is_chitchat = not any(k in prompt for k in medical_keywords) and not is_empathy and len(prompt) > 2
                                    
                                    if any(w in prompt for w in cold_words):
                                        st.session_state.trust_score -= 30
                                        reply = "（家属瞬间暴怒）：“你什么态度！你叫谁闭嘴？我要投诉你！！”"
                                        st.session_state.messages.append({"role": "assistant", "content": reply})
                                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                                        if check_medical_dispute(): st.rerun()
                                    
                                    elif is_empathy:
                                        st.session_state.trust_score = min(100, st.session_state.trust_score + 20)
                                        st.session_state.score_empathy += 10
                                        st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                                        system_prompt = build_parent_prompt(case, st.session_state.trust_score, st.session_state.disease_progress, st.session_state.unlocked_clues, is_furious)
                                        calm_prompt = system_prompt + f"\n\n【本轮特殊指令】医生刚刚安慰了你（原话：{prompt}）。你被触动了，情绪稍微平复，哽咽着回应一句，不超过 25 字。"
                                        try: 
                                            r = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "system", "content": calm_prompt}], temperature=0.9, max_tokens=40)
                                            reply = r.choices[0].message.content.strip()
                                        except Exception: reply = "（家属愣了一下，情绪稍微平复了一些）：“医生……谢谢您，求您一定救救孩子……”"
                                        st.session_state.messages.append({"role": "assistant", "content": reply})
                                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                                        
                                    elif is_chitchat:
                                        st.session_state.trust_score -= 5
                                        st.session_state.penalty_log.append("问诊时废话连篇")
                                        reply = "（家属急得直跺脚）：“医生！都火烧眉毛了您怎么还有心思问这个！求您快看看孩子吧！”"
                                        st.session_state.messages.append({"role": "assistant", "content": reply})
                                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                                        st.session_state.disease_progress += 2 
                                        if check_medical_dispute(): st.rerun()
                                    else:
                                        try:
                                            system_prompt = build_parent_prompt(case, st.session_state.trust_score, st.session_state.disease_progress, st.session_state.unlocked_clues, is_furious)
                                            api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history[-12:]
                                            response = client.chat.completions.create(model=MODEL_NAME, messages=api_messages, temperature=0.85, top_p=0.9, max_tokens=150, stream=True)
                                            
                                            message_placeholder = st.empty()
                                            ai_reply = ""
                                            for chunk in response:
                                                if chunk.choices[0].delta.content:
                                                    ai_reply += chunk.choices[0].delta.content
                                                    with message_placeholder.container(): render_message("assistant", ai_reply + "▌", case)
                                            message_placeholder.empty()
                                        except Exception as e: ai_reply = "家属显得很烦躁，没有回答你的问题。" if is_furious else "医生，您快给看看，到底怎么回事啊？"

                                        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                                        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                                        
                                        match_text = prompt + " " + ai_reply
                                        for clue in case["clues"]:
                                            if clue not in st.session_state.unlocked_clues and check_clue_trigger(clue, match_text):
                                                st.session_state.unlocked_clues.append(clue); st.session_state.score_inquiry += 8; st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                                                if is_furious: st.session_state.messages.append({"role": "assistant", "content": f"（家属正在发火，但你敏锐地捕捉到了关键信息：{clue}）"})
                                                explanation = CLUE_EXPLANATIONS.get(clue, "发现了一个关键临床体征。")
                                                st.toast(f"📖 解锁线索：{clue}\n{explanation}", icon="📚")
                                    st.rerun()

                        elif st.session_state.time_period == 2:
                            st.info("🩺 请根据目前收集到的线索，输入您的初步诊断：")
                            diag_input = st.text_input("输入疾病名称（如：急性喉炎、气道异物等）：")
                            if st.button("确认诊断，进入抢救", type="primary", use_container_width=True):
                                if not diag_input.strip(): st.error("请输入诊断结果！")
                                else:
                                    if any(k in diag_input for k in case["disease"]):
                                        st.session_state.score_diagnosis += 15
                                        st.session_state.messages.append({"role": "assistant", "content": f"✅ 诊断正确！初步考虑：{case['disease']}。"})
                                        st.session_state.disease_progress = max(50, st.session_state.disease_progress) 
                                    else:
                                        st.session_state.disease_progress = min(100, st.session_state.disease_progress + 25)
                                        st.session_state.trust_score -= 20
                                        st.session_state.penalty_log.append(f"误诊为：{diag_input}")
                                        st.session_state.messages.append({"role": "assistant", "content": f"❌ 误诊！你给出的“{diag_input}”方向错误，病情加重！"})
                                    st.session_state.diagnosis_processed = True
                                    st.session_state.time_period = 3
                                    st.session_state.action_points = 3 if st.session_state.game_mode == "easy" else 2
                                    st.session_state.crisis_start_time = time.time()
                                    st.session_state.messages.append({"role": "assistant", "content": "（就在你思考的瞬间，患儿突发严重的呼吸困难！）"})
                                    st.rerun()

                        elif st.session_state.time_period == 3:
                            if st.session_state.crisis_correct_count >= 2:
                                st.success("🎉 患儿症状明显缓解，生命体征趋于平稳！抢救成功！")
                                if st.button("✅ 抢救成功，进入结案复盘", key="crisis_success_center", use_container_width=True, type="primary"):
                                    st.session_state.score_emergency += 20
                                    st.session_state.time_period = 4
                                    st.session_state.game_over = True
                                    st.session_state.disease_progress = min(40, max(10, st.session_state.disease_progress))
                                    st.rerun()
                                return 
                                
                            elapsed = time.time() - st.session_state.crisis_start_time
                            time_left = max(0, 90 - int(elapsed))
                            
                            # ================= 真实的倒计时 =================
                            if time_left > 0:
                                st.markdown(f'''
                                    <div style="background-color: #fee2e2; color: #ef4444; padding: 15px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 18px; margin-bottom: 15px;">
                                        ⏰ 紧急抢救倒计时：<span id="timer">{time_left}</span> 秒！请立即选择：
                                    </div>
                                    <script>
                                        var timeleft = {time_left};
                                        var downloadTimer = setInterval(function(){{
                                            timeleft -= 1;
                                            if(timeleft <= 0){{
                                                clearInterval(downloadTimer);
                                                document.getElementById("timer").innerHTML = "0";
                                                document.getElementById("timeout_trigger").click();
                                            }} else {{
                                                document.getElementById("timer").innerHTML = timeleft;
                                            }}
                                        }}, 1000);
                                    </script>
                                ''', unsafe_allow_html=True)
                                
                                # 隐藏超时按钮
                                st.markdown("""<style>div[data-testid="stButton"]:has(button[key="timeout_trigger"]) {display: none;}</style>""", unsafe_allow_html=True)
                                if st.button("timeout", key="timeout_trigger"):
                                    st.session_state.disease_progress = 100
                                    st.session_state.game_over = True
                                    st.session_state.penalty_log.append("抢救超时，未能及时干预")
                                    st.rerun()
                            else:
                                st.session_state.disease_progress = 100
                                st.session_state.game_over = True
                                st.session_state.penalty_log.append("抢救超时")
                                st.rerun()
                            # ====================================================

                            if st.session_state.action_points > 0:
                                for opt_key, opt_val in case["crisis"].items():
                                    if st.button(opt_val["label"], key=f"crisis_{opt_key}", use_container_width=True):
                                        st.session_state.action_points -= 1
                                        st.session_state.messages.append({"role": "user", "content": opt_val["label"]})
                                        st.session_state.messages.append({"role": "assistant", "content": opt_val["reply"]})
                                        st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + opt_val["disease"]))
                                        
                                        time_penalty = 5 if opt_val["type"] == "correct" else 10
                                        st.session_state.crisis_start_time -= time_penalty
                                        
                                        if opt_val["type"] == "correct": st.session_state.crisis_correct_count += 1
                                        else:
                                            st.session_state.penalty_log.append(opt_val["label"])
                                            st.session_state.trust_score -= 25
                                            if st.session_state.disease_progress >= 100: st.session_state.game_over = True
                                            if check_medical_dispute(): st.rerun()
                                        st.rerun() 
                            else:
                                st.warning("⚠️ 你的体力已耗尽，无法进行更多操作！等待命运裁决...")
                                if st.button("等待结果"):
                                    if st.session_state.crisis_correct_count >= 2:
                                        st.session_state.time_period = 4
                                        st.session_state.game_over = True
                                        st.session_state.disease_progress = max(10, st.session_state.disease_progress)
                                    else:
                                        st.session_state.disease_progress = 100
                                        st.session_state.game_over = True
                                    st.rerun()
            chat_interface()
            
    # ================= 第四幕：结案复盘 =================
    if st.session_state.game_over:
        st.divider(); st.header("🩺 带教老师复盘")
        if st.session_state.final_score == 0:
            total = (st.session_state.score_inquiry + st.session_state.score_diagnosis + st.session_state.score_emergency + st.session_state.score_empathy)
            st.session_state.final_score = total
            if total >= 80: st.session_state.final_title = "🏆 儿科急诊之光"
            elif total >= 60: st.session_state.final_title = "🌟 有潜力的住院医"
            else: st.session_state.final_title = "📚 还需回炉重造"

        if not st.session_state.record_saved:
            check_achievements(case, st.session_state.current_case)
            save_record({"player_id": st.session_state.player_id, "player_name": st.session_state.player_name, "case_id": st.session_state.current_case, "case_name": case["name"], "disease": case["disease"], "mode": st.session_state.game_mode, "final_score": st.session_state.final_score, "score_inquiry": st.session_state.score_inquiry, "score_diagnosis": st.session_state.score_diagnosis, "score_emergency": st.session_state.score_emergency, "score_empathy": st.session_state.score_empathy, "disease_progress": st.session_state.disease_progress, "survived": 1 if st.session_state.disease_progress < 100 else 0, "mistakes": json.dumps(st.session_state.penalty_log, ensure_ascii=False), "achievements": json.dumps(st.session_state.achievements, ensure_ascii=False), "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.session_state.record_saved = True

        if not st.session_state.ai_review_generated:
            with st.spinner("👨‍⚕️ 主治医师正在整理疾病知识点并进行查房复盘..."):
                try:
                    missed_clues = set(case['clues']) - set(st.session_state.unlocked_clues)
                    all_clues = "、".join(case['clues'])
                    review_prompt = f"""
                    你是一名严厉但负责的三甲医院儿科主任医师。正在进行临床带教查房复盘。你正在当面点评刚刚接诊完【{case['disease']}】的住院医（玩家）。
                    【本病例标准核心线索】：{all_clues}
                    【玩家漏掉的线索】：{missed_clues}
                    【玩家的操作失误】：{st.session_state.penalty_log}
                    【核心要求】：必须全程使用第二人称“你”！绝对禁止使用“他”、“该医生”等第三人称。
                    请用 250 字左右的医学专业口吻进行点评，必须严格按以下三个部分输出：
                    1. 【疾病知识微课】：先花两句话简明扼要地讲解【{case['disease']}】的典型临床表现是什么，以及接诊时【最核心的问诊/查体要点】应该包含哪些。
                    2. 【你的实战评价】：结合本病例的标准线索，一针见血地指出玩家刚才的表现（比如：你问出了什么、漏掉了什么、思维哪里不到位）。
                    3. 【临床干预红线】：强调该病在急救或处置上绝对禁止的操作（避坑），以及首选的抢救原则。
                    （如果玩家完美通关，也要简要复习知识点并予以肯定）。
                    """
                    res = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": review_prompt}], temperature=0.6)
                    st.session_state.ai_review = res.choices[0].message.content
                    st.session_state.ai_review_generated = True
                except: st.session_state.ai_review = "主任今天太忙，未能给出评语。不过，多复盘永远是进步的阶梯。"; st.session_state.ai_review_generated = True
        
        st.info(f"### {st.session_state.final_title}\n最终得分：{st.session_state.final_score} / 100\n\n**👨‍⚕️ 主治医师点评：**\n{st.session_state.ai_review}")

        fig = go.Figure(go.Scatterpolar(r=[min(100, st.session_state.score_inquiry * 3), min(100, st.session_state.score_diagnosis * 2), min(100, st.session_state.score_emergency * 2), min(100, st.session_state.score_empathy * 2)], theta=['问诊/查体', '诊断', '急救', '沟通/人文'], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🎓 生成专属虚拟医师执照")
        if st.button("立即生成证书"):
            try:
                W, H = 800, 1100
                img = Image.new('RGB', (W, H), color=(255, 255, 255))
                d = ImageDraw.Draw(img)
                d.rectangle([15, 15, W-15, H-15], outline=(100, 150, 200), width=6)
                d.rectangle([25, 25, W-25, H-25], outline=(150, 200, 230), width=2)
                
                font_paths = ["/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", "C:/Windows/Fonts/msyh.ttc", "/System/Library/Fonts/PingFang.ttc"]
                font_title = font_text = font_bold = None
                for fp in font_paths:
                    if os.path.exists(fp):
                        font_title = ImageFont.truetype(fp, 50)
                        font_text = ImageFont.truetype(fp, 28)
                        font_bold = ImageFont.truetype(fp, 32)
                        break
                if not font_title: font_title = font_text = font_bold = ImageFont.load_default()

                d.text((W//2, 100), "虚拟医师执业证书", fill=(0, 0, 0), font=font_title, anchor="mm")
                d.text((W//2, 160), "PediMind Medical Certificate", fill=(150, 150, 150), font=font_text, anchor="mm")
                
                info_x, start_y, line_gap = 80, 260, 55
                info_data = [("姓　　名：", st.session_state.player_name), ("执业级别：", "执业医师"), ("任职资格：", st.session_state.final_title.replace("🏆 ", "").replace("🌟 ", "").replace("📚 ", "")), ("病例挑战：", case['name']), ("最终得分：", f"{st.session_state.final_score} 分"), ("证书编号：", f"Pedi-{str(uuid.uuid4())[:8].upper()}")]
                for i, (label, value) in enumerate(info_data):
                    y = start_y + i * line_gap
                    d.text((info_x, y), label, fill=(80, 80, 80), font=font_text)
                    d.text((info_x + 180, y), str(value), fill=(0, 0, 0), font=font_bold)
                
                if os.path.exists("static/doctor.png"):
                    avatar = Image.open("static/doctor.png").convert("RGBA").resize((150, 150))
                    img.paste(avatar, (W - 250, 200), avatar)
                
                qr = qrcode.make(f"http://localhost:8501/?pid={st.session_state.player_id}").resize((160, 160))
                img.paste(qr, (W // 2 - 80, 750))
                
                d.text((W//2, 950), "签发机关：PediMind 虚拟医学教育中心", fill=(50, 50, 50), font=font_text, anchor="mm")
                d.text((W//2, 1000), f"签发日期：{datetime.now().strftime('%Y年%m月%d日')}", fill=(50, 50, 50), font=font_text, anchor="mm")

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.image(byte_im, caption="长按或右键保存你的专属证书", use_container_width=True)
                st.download_button(label="📥 下载高清证书", data=byte_im, file_name=f"PediMind_Cert_{st.session_state.player_name}.png", mime="image/png")
            except Exception as e:
                st.error(f"证书生成失败：{e}")

        if st.button("🏠 返回主菜单"):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "player_id", "player_name", "achievements", "completed_cases"]: del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()
