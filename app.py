import streamlit as st
import os
import time
import json
import json as _json
import random
import base64
import sqlite3
import uuid
import html
import re
from datetime import datetime
from contextlib import contextmanager
import pandas as pd
from openai import OpenAI
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

# ================= 1. 配置区 =================
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

try:
    API_KEY = st.secrets["ZHIPU_API_KEY"]
except Exception:
    API_KEY = "sk-你的真实智谱AI密钥"  # 替换为你的Key

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 2. 四大病例数据库 =================
CASES_CONFIG = {
    "febrile_seizure": {
        "name": "2岁患儿的突发抽搐", "disease": "热性惊厥", "patient": "豆豆，2岁，男", "difficulty": "⭐⭐⭐",
        "description": "发热后突发意识丧失、四肢抽搐，家长极度恐慌，抱着孩子冲进急诊。",
        "audio": "none.mp3", "images": {},
        "clues": ["发热史", "突发抽搐", "双眼上翻", "意识丧失", "自行缓解"],
        "ai_role": "2岁热性惊厥患儿的妈妈。极度恐慌、哭泣。",
        "ai_rules": "1.只说大白话(如'翻白眼'、'抽抽'、'我掐人中了')。2.禁止给诊断。3.50字以内。",
        "diagnosis": {"A": "癫痫发作", "B": "热性惊厥", "C": "化脓性脑膜炎", "D": "低血糖"},
        "crisis": {
            "A": {"label": "保持侧卧位，清理口鼻分泌物，防止跌落", "type": "correct", "disease": -30, "reply": "（患儿抽搐逐渐停止，呼吸平稳）"},
            "B": {"label": "强行按压四肢，往嘴里塞毛巾防止咬舌", "type": "harmful", "disease": 40, "reply": "（家长强行按压导致患儿骨折，毛巾堵塞气道引发窒息）"},
            "C": {"label": "用力掐人中、虎口，试图唤醒患儿", "type": "invalid", "disease": 10, "reply": "（掐人中无效，反而造成局部皮肤破损）"},
            "D": {"label": "立即给予静脉推注地西泮止痉", "type": "overuse", "disease": 15, "reply": "（患儿已自行缓解，过度用药导致呼吸抑制）"}
        }
    },
    "intussusception": {
        "name": "8月龄婴儿的阵发哭闹", "disease": "急性肠套叠", "patient": "壮壮，8个月，男", "difficulty": "⭐⭐⭐⭐",
        "description": "婴儿突发阵发性剧烈哭闹，伴呕吐，家长以为是吃坏了肚子。",
        "audio": "none.mp3", "images": {},
        "clues": ["阵发性哭闹", "屈腿缩腹", "呕吐", "果酱样血便", "腹部包块"],
        "ai_role": "8个月肠套叠患儿的妈妈。焦急、疲惫。",
        "ai_rules": "1.说大白话(如'一阵阵哭得撕心裂肺'、'拉了点红色的像果酱')。2.禁止给诊断。3.50字以内。",
        "diagnosis": {"A": "急性胃肠炎", "B": "肠痉挛", "C": "急性肠套叠", "D": "阑尾炎"},
        "crisis": {
            "A": {"label": "禁食水，胃肠减压，立即准备空气灌肠", "type": "correct", "disease": -30, "reply": "（空气灌肠复位成功，患儿安静入睡）"},
            "B": {"label": "给予强效止痛药缓解哭闹", "type": "harmful", "disease": 35, "reply": "（止痛药掩盖了病情，延误抢救导致肠坏死）"},
            "C": {"label": "腹部热敷并顺时针按摩", "type": "harmful", "disease": 40, "reply": "（热敷加重了肠道充血，导致肠穿孔）"},
            "D": {"label": "喂服温水和口服补液盐", "type": "invalid", "disease": 20, "reply": "（患儿呕吐加剧，误吸导致吸入性肺炎）"}
        }
    },
    "anaphylaxis": {
        "name": "5岁患儿的致命红疹", "disease": "过敏性休克", "patient": "米米，5岁，女", "difficulty": "⭐⭐⭐⭐⭐",
        "description": "进食坚果后突发全身皮疹、呼吸急促、面色苍白，生死一线。",
        "audio": "wheezing.mp3", "images": {},
        "clues": ["进食坚果史", "全身风团皮疹", "嘴唇肿胀", "呼吸困难", "面色苍白"],
        "ai_role": "5岁过敏休克患儿的妈妈。极度恐慌、语无伦次。",
        "ai_rules": "1.说大白话(如'全身起大红疹子'、'嘴唇肿得像香肠')。2.禁止给诊断。3.50字以内。",
        "diagnosis": {"A": "急性荨麻疹", "B": "食物中毒", "C": "过敏性休克", "D": "支气管哮喘"},
        "crisis": {
            "A": {"label": "立即大腿外侧肌注肾上腺素，平卧抬高下肢", "type": "correct", "disease": -30, "reply": "（血压回升，皮疹消退，呼吸恢复平稳）"},
            "B": {"label": "静脉推注地塞米松和葡萄糖酸钙", "type": "overuse", "disease": 20, "reply": "（激素起效太慢，患儿气道水肿加重，出现窒息）"},
            "C": {"label": "喂水催吐，试图排出过敏原", "type": "harmful", "disease": 40, "reply": "（催吐导致误吸，患儿呼吸骤停）"},
            "D": {"label": "坐位高流量吸氧，等待救护车", "type": "invalid", "disease": 15, "reply": "（坐位加重了血压下降，等待过程中休克恶化）"}
        }
    },
    "dka": {
        "name": "6岁患儿的深大呼吸", "disease": "糖尿病酮症酸中毒", "patient": "小杰，6岁，男", "difficulty": "⭐⭐⭐⭐⭐",
        "description": "近期多饮多尿，今日突发呕吐、腹痛、深大呼吸，逐渐陷入昏迷。",
        "audio": "none.mp3", "images": {},
        "clues": ["多饮多尿", "近期消瘦", "呼气烂苹果味", "深大呼吸", "精神萎靡"],
        "ai_role": "6岁DKA患儿的妈妈。焦急、不知所措。",
        "ai_rules": "1.说大白话(如'最近喝水尿尿特别多'、'嘴里一股奇怪的烂苹果味')。2.禁止给诊断。3.50字以内。",
        "diagnosis": {"A": "急性胃肠炎", "B": "中枢神经系统感染", "C": "糖尿病酮症酸中毒", "D": "尿崩症"},
        "crisis": {
            "A": {"label": "建立双通道静脉补液，小剂量胰岛素持续静滴", "type": "correct", "disease": -30, "reply": "（血糖平稳下降，酸中毒逐渐纠正，患儿苏醒）"},
            "B": {"label": "大剂量胰岛素快速静推降糖", "type": "harmful", "disease": 40, "reply": "（血糖下降过快，引发脑水肿，患儿抽搐昏迷）"},
            "C": {"label": "立即静脉推注高渗葡萄糖", "type": "harmful", "disease": 35, "reply": "（加重了高渗状态，病情急剧恶化）"},
            "D": {"label": "给予碳酸氢钠快速纠正酸中毒", "type": "overuse", "disease": 20, "reply": "（过度纠酸导致反常性脑脊液酸中毒，加重脑损伤）"}
        }
    }
}

# ================= 2.5 AI 对话层与线索判定层 =================
CLUE_SYNONYMS = {
    "发热史":         ["发烧", "发热", "烧了", "体温高"],
    "突发抽搐":       ["抽抽", "抽搐", "抽筋", "抽风", "全身抖动"],
    "双眼上翻":       ["翻白眼", "眼睛往上翻", "眼神不对"],
    "意识丧失":       ["叫不醒", "没反应", "不答应", "意识没了"],
    "自行缓解":       ["停下来了", "现在好了", "自己停了"],
    
    "阵发性哭闹":     ["一阵阵哭", "撕心裂肺", "怎么哄都不行"],
    "屈腿缩腹":       ["腿缩着", "蜷着腿", "缩着肚子"],
    "呕吐":           ["吐了", "吐奶", "吃什么吐什么"],
    "果酱样血便":     ["果酱", "红粑粑", "红色的便", "带血的粑粑"],
    "腹部包块":       ["肚子有硬块", "摸肚子有包", "肚子鼓包"],
    
    "进食坚果史":     ["吃坚果", "吃花生", "吃了腰果"],
    "全身风团皮疹":   ["起红疹", "大红疹子", "风团", "起包"],
    "嘴唇肿胀":       ["嘴唇肿", "香肠嘴", "嘴巴肿了"],
    "面色苍白":       ["脸色白", "脸惨白", "没血色"],
    "呼吸困难":       ["喘不上气", "呼吸困难", "憋气", "呼吸急促"],
    
    "多饮多尿":       ["喝水多", "尿多", "尿特别多", "老口渴"],
    "近期消瘦":       ["瘦了一大圈", "瘦了", "体重下降"],
    "呼气烂苹果味":   ["烂苹果", "奇怪的味", "嘴里有味"],
    "深大呼吸":       ["喘气特别大", "呼吸很大声", "大喘气"],
    "精神萎靡":       ["叫不醒", "没精神", "软绵绵"],
}

BANNED_TERMS = [
    "急性", "喉炎", "会厌", "哮喘", "异物", "支气管", "诊断", "确诊",
    "惊厥", "癫痫", "肠套叠", "休克", "酮症", "酸中毒", "糖尿病", "脑膜炎",
    "三凹征", "哮鸣", "发绀", "Stridor", "Wheezing", "糖皮质激素", "肾上腺素"
]

MOM_FACTS = {
    "febrile_seizure": [
        "刚才发烧，突然就翻白眼了，抽抽，吓死我了",
        "叫他也不答应，整个人软软的",
        "我赶紧给他掐人中，现在好像停下来了",
        "以前从来没这样过，是不是烧坏脑子了啊医生"
    ],
    "intussusception": [
        "一阵一阵哭得撕心裂肺，怎么哄都不行，腿还缩着",
        "吃奶全吐了，刚才拉了点像果酱一样的红粑粑",
        "摸他肚子好像有个硬块，一摸就哭得更厉害",
        "刚开始还以为是吃坏肚子了"
    ],
    "anaphylaxis": [
        "刚才吃了点坚果，突然全身起大红疹子，一大片一大片的",
        "嘴唇肿得像香肠一样，眼睛也肿了",
        "呼吸很急，听着嗓子里有声音，脸色惨白",
        "他以前对花生好像有点过敏，今天没防住"
    ],
    "dka": [
        "最近一个月喝水特别多，尿也特别多，瘦了一大圈",
        "今天一直说肚子疼，还吐了好几次",
        "刚才看着他喘气特别大，呼哧呼哧的",
        "嘴里有一股奇怪的味道，像烂苹果，怎么叫都叫不醒"
    ],
}

FALLBACK_REPLIES = {
    "突发抽搐":     "他刚才突然抽抽，翻白眼，吓死我了…",
    "阵发性哭闹":   "他一阵一阵哭得撕心裂肺，怎么哄都不行…",
    "全身风团皮疹": "突然全身起大红疹子，嘴唇也肿了…",
    "深大呼吸":     "他喘气特别大，怎么叫都叫不醒…",
    "default":      "医生，我真的好担心，求您快帮帮孩子吧！",
}

def _ai_available() -> bool:
    return bool(API_KEY) and "你的" not in str(API_KEY) and len(str(API_KEY)) > 20

def match_clues(prompt: str, case: dict) -> list:
    text = prompt.lower()
    hits = []
    for clue in case["clues"]:
        synonyms = [clue] + CLUE_SYNONYMS.get(clue, [])
        if any(s.lower() in text for s in synonyms):
            hits.append(clue)
    return hits

def sanitize_reply(reply):
    if not isinstance(reply, str): return None
    reply = reply.strip()
    if not reply or len(reply) > 80: return None
    if any(t in reply for t in BANNED_TERMS): return None
    return reply

def pick_fallback(hit_clues: list) -> str:
    for c in hit_clues:
        if c in FALLBACK_REPLIES: return FALLBACK_REPLIES[c]
    return FALLBACK_REPLIES["default"]

def build_mom_system(case_id: str, case: dict) -> str:
    facts = MOM_FACTS.get(case_id, [])
    fact_lines = "\n".join(f"- {f}" for f in facts)
    return f"""你是一位{case['patient']}的妈妈，此刻在急诊室，非常焦虑。
你只知道自己亲眼看到、亲耳听到的事情（见下），你完全不懂医学，从没听过任何疾病名称。
【你知道的事实】
{fact_lines}
【说话规则】
1. 只用大白话，像普通家长一样说话，绝对不说任何医学名词、病名、检查名、药名。
2. 绝不下诊断，绝不说"是不是XX病"。
3. 每次只说 1~2 句话，不超过50个字，语气焦虑、无助。
4. 医生问到你不知道的，就说"这我也不懂啊医生"。
5. 只输出 JSON：{{"reply": "你的回答"}}，不要输出其他任何内容。"""

def _clean_history(history: list) -> list:
    out = []
    for m in history:
        c = re.sub(r"（提示：[^）]*）", "", m.get("content", "")).strip()
        if c:
            out.append({"role": "user" if m.get("role") == "user" else "assistant", "content": c})
    return out

def call_mom_llm(case_id: str, case: dict, history: list, hit_clues: list):
    if not _ai_available(): return None
    system = build_mom_system(case_id, case)
    msgs = [{"role": "system", "content": system}] + _clean_history(history[-6:])
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=msgs, temperature=0.8,
                max_tokens=120, timeout=6, response_format={"type": "json_object"}
            )
            data = _json.loads(resp.choices[0].message.content)
            reply = sanitize_reply(data.get("reply"))
            if reply: return reply
        except Exception:
            continue
    return None

# ================= 3. 成就系统 =================
ACHIEVEMENTS = {
    "first_inquiry": {"icon": "🩺", "name": "初次问诊", "desc": "迈出问诊第一步"},
    "all_clues":     {"icon": "🔍", "name": "明察秋毫", "desc": "集齐全部线索"},
    "correct_dx":    {"icon": "🎯", "name": "一针见血", "desc": "初步诊断完全正确"},
    "lifesaver":     {"icon": "⚡", "name": "妙手回春", "desc": "急救处置正确 ≥2 次"},
    "empathy":       {"icon": "💬", "name": "医者仁心", "desc": "用共情方式交代病情"},
    "perfect":       {"icon": "🏆", "name": "急诊之光", "desc": "最终得分 ≥ 80"},
}

def unlock_achievement(aid):
    if aid not in st.session_state.achievements:
        st.session_state.achievements.append(aid)

def check_achievements(case):
    ss = st.session_state
    if any(m.get("role") == "user" for m in ss.messages): unlock_achievement("first_inquiry")
    if len(ss.unlocked_clues) >= len(case["clues"]): unlock_achievement("all_clues")
    if ss.diagnosis_made and case["diagnosis"].get(ss.diagnosis_made) == case["disease"]: unlock_achievement("correct_dx")
    if ss.crisis_correct_count >= 2: unlock_achievement("lifesaver")
    if ss.score_empathy > 0: unlock_achievement("empathy")
    if ss.final_score >= 80: unlock_achievement("perfect")

def render_achievements(container=st.sidebar):
    container.markdown("### 🏅 成就墙")
    cols = container.columns(3)
    for i, (aid, a) in enumerate(ACHIEVEMENTS.items()):
        unlocked = aid in st.session_state.get("achievements", [])
        with cols[i % 3]:
            icon_style = "" if unlocked else "filter:grayscale(1) opacity(0.35);"
            bg     = "rgba(0,180,216,0.15)" if unlocked else "rgba(128,128,128,0.08)"
            border = "#00b4d8" if unlocked else "rgba(150,150,150,0.35)"
            name_color = "#00b4d8" if unlocked else "#8a94a6"
            st.markdown(f"""
            <div title="{a['desc']}" style="text-align:center; background:{bg}; border:1px solid {border};
                        border-radius:12px; padding:10px 6px; margin-bottom:8px;">
                <div style="font-size:26px; {icon_style}">{a['icon']}</div>
                <div style="font-size:12px; font-weight:700; color:{name_color}; margin-top:4px;">{a['name']}</div>
            </div>""", unsafe_allow_html=True)

# ================= 4. 数据持久化层 =================
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
            final_score INTEGER, score_inquiry INTEGER, score_diagnosis INTEGER,
            score_emergency INTEGER, score_empathy INTEGER,
            disease_progress INTEGER, survived INTEGER,
            mistakes TEXT, achievements TEXT, created_at TEXT
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

def fetch_my_records(player_name: str):
    with db() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM records WHERE player_name=? ORDER BY id DESC", (player_name,)).fetchall()
    return [dict(r) for r in rows]

def fetch_all_records():
    with db() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

# ================= 5. 头像 & 消息气泡 =================
@st.cache_data(show_spinner=False)
def load_avatar(path: str) -> str:
    if not os.path.exists(path): return ""
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
            return f'<img src="{src}" style="width:{AVATAR_SIZE}; height:{AVATAR_SIZE}; border-radius:50%; object-fit:cover; flex-shrink:0; box-shadow:0 1px 3px rgba(0,0,0,0.2);">'
        return f'<div style="width:{AVATAR_SIZE}; height:{AVATAR_SIZE}; border-radius:50%; background:{fallback_bg}; color:#fff; display:flex; align-items:center; justify-content:center; flex-shrink:0; font-size:18px;">{fallback_emoji}</div>'

    if role == "user":
        bg = "#95ec69" if not night else "#3eb575"
        fg = "#111111" if not night else "#ffffff"
        av = avatar_html("doctor.png", "👨‍⚕️", "#4a90d9")
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
        av = avatar_html("mom.png", "👩", "#f6a623")
        st.markdown(f"""
        <div style="display:flex; justify-content:flex-start; align-items:flex-start; margin:8px 0;">
            <div style="margin-right:8px;">{av}</div>
            <div style="background:{bg}; color:{fg}; padding:10px 14px; border-radius:14px;
                        border-top-left-radius:3px; max-width:75%; word-break:break-word;
                        font-size:15px; line-height:1.6; box-shadow:0 1px 2px rgba(0,0,0,0.12);">
                {safe}
            </div>
        </div>""", unsafe_allow_html=True)

# ================= 6. 专业四通道重症监护仪 =================
def render_monitor(disease_progress: int):
    if disease_progress < 40:
        hr, spo2, rr, sys, dia, status = 120, 97, 28, 100, 60, "normal"
    elif disease_progress < 70:
        hr, spo2, rr, sys, dia, status = 140, 93, 35, 90, 50, "warn"
    elif disease_progress < 90:
        hr, spo2, rr, sys, dia, status = 160, 88, 45, 80, 40, "danger"
    else:
        hr, spo2, rr, sys, dia, status = 180, 82, 55, 70, 30, "critical"

    html_code = f"""
    <!DOCTYPE html><html><head><style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      body {{ background:#000; overflow:hidden; font-family: 'Courier New', monospace; }}
      .monitor {{ display: flex; width:100%; height:240px; background:#000; border-radius:8px; border:1px solid #333; overflow:hidden; }}
      .canvas-container {{ flex: 1; position: relative; background:#000; }}
      canvas {{ display: block; width: 100%; height: 100%; }}
      .vitals-panel {{ width: 120px; display: flex; flex-direction: column; justify-content: space-around; background: #050505; padding-right: 12px; border-left: 1px solid #1a1a1a; }}
      .vital-row {{ display: flex; flex-direction: column; align-items: flex-end; line-height: 1; }}
      .vital-top {{ display: flex; align-items: flex-end; gap: 4px; }}
      .v-label {{ font-size: 12px; font-weight: bold; margin-bottom: 2px; }}
      .v-num {{ font-size: 32px; font-weight: bold; letter-spacing: -1px; }}
      .v-unit {{ font-size: 10px; color: #555; margin-top: 2px; }}
      .blink {{ animation: blink 0.6s infinite; }}
      @keyframes blink {{ 50% {{ opacity: 0.3; }} }}
    </style></head><body>
      <div class="monitor">
        <div class="canvas-container"><canvas id="wave-canvas"></canvas></div>
        <div class="vitals-panel">
            <div class="vital-row"><div class="vital-top"><div class="v-label" style="color:#00ff00">HR</div><div class="v-num" style="color:#00ff00" id="v-hr">--</div></div><div class="v-unit">bpm</div></div>
            <div class="vital-row"><div class="vital-top"><div class="v-label" style="color:#ff3333">ABP</div><div class="v-num" style="color:#ff3333; font-size: 24px;" id="v-abp">--/--</div></div><div class="v-unit">mmHg</div></div>
            <div class="vital-row"><div class="vital-top"><div class="v-label" style="color:#ffff00">SpO₂</div><div class="v-num" style="color:#ffff00" id="v-spo2">--</div></div><div class="v-unit">%</div></div>
            <div class="vital-row"><div class="vital-top"><div class="v-label" style="color:#ffffff">RESP</div><div class="v-num" style="color:#ffffff" id="v-rr">--</div></div><div class="v-unit">/min</div></div>
        </div>
      </div>
    <script>
        const HR={hr}, SPO2={spo2}, RR={rr}, SYS={sys}, DIA={dia}, STATUS="{status}";
        const canvas = document.getElementById('wave-canvas');
        const ctx = canvas.getContext('2d');
        let W, H, pxPerSec = 120; 
        let t = 0, lastTime = performance.now();

        function resize() {{
            const r = canvas.parentElement.getBoundingClientRect();
            canvas.width = r.width * devicePixelRatio; canvas.height = r.height * devicePixelRatio;
            ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
            W = r.width; H = r.height; pxPerSec = W / 2.5;
        }}
        window.addEventListener('resize', resize); resize();

        function getECG(p) {{ if(p<0)p+=1; let v = 0.1*Math.exp(-Math.pow((p-0.15)/0.02,2)) - 0.2*Math.exp(-Math.pow((p-0.24)/0.008,2)) + 1.0*Math.exp(-Math.pow((p-0.25)/0.010,2)) - 0.3*Math.exp(-Math.pow((p-0.27)/0.010,2)) + 0.25*Math.exp(-Math.pow((p-0.4)/0.04,2)); return v; }}
        function getABP(p) {{ if(p<0)p+=1; let v = 1.0*Math.exp(-Math.pow((p-0.15)/0.03,2)) + 0.4*Math.exp(-Math.pow((p-0.4)/0.15,2)) + 0.1*Math.exp(-Math.pow((p-0.6)/0.05,2)); return Math.max(0,v); }}
        function getSpO2(p) {{ if(p<0)p+=1; return Math.max(0, Math.sin(p*Math.PI*2) + 0.2*Math.sin(p*Math.PI*4)); }}
        function getResp(p) {{ if(p<0)p+=1; return Math.sin(p*Math.PI*2); }}

        function drawChannel(yBase, yHeight, dataFunc, phaseMult, color, lineWidth) {{
            ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.lineJoin = 'round';
            for (let x = 0; x < W; x++) {{
                let timeAgo = (W - x) / pxPerSec;
                let phase = ((t - timeAgo) * phaseMult) % 1;
                let val = dataFunc(phase);
                let y = yBase - val * yHeight;
                if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }}
            ctx.stroke();
        }}

        function draw() {{
            ctx.fillStyle = "#000"; ctx.fillRect(0, 0, W, H);
            const chHeight = H / 4;
            ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
            for(let i=1; i<4; i++) {{ ctx.beginPath(); ctx.moveTo(0, chHeight*i); ctx.lineTo(W, chHeight*i); ctx.stroke(); }}
            let hrMult = HR / 60; let rrMult = RR / 60;
            drawChannel(chHeight*0.75, chHeight*0.5, getECG, hrMult, "#00ff00", 1.5);
            drawChannel(chHeight*1.85, chHeight*0.5, getABP, hrMult, "#ff3333", 1.5);
            drawChannel(chHeight*2.85, chHeight*0.5, getSpO2, hrMult, "#ffff00", 1.5);
            drawChannel(chHeight*3.65, chHeight*0.4, getResp, rrMult, "#ffffff", 1.5);
        }}

        function loop(now) {{
            let dt = (now - lastTime) / 1000; lastTime = now;
            if (dt > 0.1) dt = 0.1; t += dt; draw(); requestAnimationFrame(loop);
        }}

        function jit(base, amt) {{ return Math.round(base + (Math.random()-0.5)*amt); }}
        function updateVitals() {{
            let hrEl = document.getElementById('v-hr');
            hrEl.textContent = jit(HR, 4);
            document.getElementById('v-abp').textContent = jit(SYS, 3) + "/" + jit(DIA, 2);
            document.getElementById('v-spo2').textContent = jit(SPO2, 1);
            document.getElementById('v-rr').textContent = jit(RR, 2);
            if (STATUS === "critical" || STATUS === "danger") hrEl.classList.add('blink'); else hrEl.classList.remove('blink');
        }}

        updateVitals(); setInterval(updateVitals, 1500); requestAnimationFrame(loop);
    </script></body></html>
    """
    components.html(html_code, height=260, scrolling=False)

# ================= 7. 页面状态与路由 =================
st.set_page_config(page_title="PediMind · 急诊风云", page_icon="🏥", layout="wide")
if "theme" not in st.session_state: st.session_state.theme = "night"
if "screen" not in st.session_state: st.session_state.screen = "start"
if "current_case" not in st.session_state: st.session_state.current_case = None
if "game_mode" not in st.session_state: st.session_state.game_mode = None
if "player_id" not in st.session_state: st.session_state.player_id = str(uuid.uuid4())[:8]
if "player_name" not in st.session_state: st.session_state.player_name = "实习医生"
if "achievements" not in st.session_state: st.session_state.achievements = []

if st.session_state.theme == "night":
    theme_css = """<style>
        .stApp { background: radial-gradient(circle at 50% 0%, #1b263b 0%, #0d1b2a 70%) !important; color: #e2e8f0 !important; }
        p, span, div, label, h1, h2, h3, h4, h5, h6 { color: #e2e8f0 !important; }
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%) !important; }
        div[data-testid="stChatMessage"] { background-color: rgba(255, 255, 255, 0.08) !important; border-radius: 12px; padding: 15px; margin-bottom: 10px; }
        div[data-testid="stChatInput"] { background-color: #ffffff !important; border-radius: 12px !important; }
        div[data-testid="stChatInput"] textarea { color: #4a5568 !important; }
        .stButton>button { border-radius: 12px !important; border: 1px solid rgba(0, 180, 216, 0.4) !important; background: rgba(0, 180, 216, 0.1) !important; color: #e2e8f0 !important; }
        .stButton>button:hover { background: rgba(0, 180, 216, 0.3) !important; }
        .status-badge { background: rgba(255, 255, 255, 0.05); border-radius: 14px; padding: 15px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); }
        .case-card { padding: 25px; border-radius: 16px; margin-bottom: 20px; background: rgba(0, 180, 216, 0.05); border: 1px solid rgba(0, 180, 216, 0.3); }
        .case-card:hover { transform: translateY(-5px); box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
        div[data-testid="stAlert"] * { color: #1a202c !important; }
    </style>"""
else:
    theme_css = """<style>
        .stApp { background-color: #f0f4f8 !important; color: #1a202c !important; }
        p, span, div, label, h1, h2, h3, h4, h5, h6 { color: #1a202c !important; }
        section[data-testid="stSidebar"] { background-color: #ffffff !important; }
        div[data-testid="stChatMessage"] { background-color: #ffffff !important; border-radius: 12px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 10px; }
        .stButton>button { border-radius: 12px !important; border: 1px solid #0077b6 !important; background: #ffffff !important; color: #0077b6 !important; }
        .status-badge { background: #ffffff; border-radius: 14px; padding: 15px; text-align: center; border: 1px solid #e2e8f0; }
        .case-card { padding: 25px; border-radius: 16px; margin-bottom: 20px; background: #ffffff; border: 1px solid #e2e8f0; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
    </style>"""
st.markdown(theme_css, unsafe_allow_html=True)

st.sidebar.markdown("### 🎨 主题设置")
if st.sidebar.button("☀️ 白天模式" if st.session_state.theme == "night" else "🌙 夜晚模式"):
    st.session_state.theme = "day" if st.session_state.theme == "night" else "night"; st.rerun()

# ================= 8. 界面渲染 =================
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
        st.text_input("👨‍⚕️ 你的昵称（用于记录错题本）", key="player_name", max_chars=12)
        if st.button("🚀 开始游戏", use_container_width=True, type="primary"):
            st.session_state.screen = "case_select"; st.rerun()
        c1, c2 = st.columns(2)
        if c1.button("📕 我的错题本", use_container_width=True): st.session_state.screen = "profile"; st.rerun()
        if c2.button("📊 全服统计", use_container_width=True): st.session_state.screen = "stats"; st.rerun()
    st.stop()

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
            if st.button(f"▶️ 进入病例", key=f"select_{case_id}", use_container_width=True, type="primary"):
                st.session_state.current_case = case_id; st.session_state.screen = "difficulty_select"; st.rerun()
    if st.button("⬅️ 返回主菜单"): st.session_state.screen = "start"; st.rerun()
    st.stop()

if st.session_state.screen == "difficulty_select":
    case = CASES_CONFIG[st.session_state.current_case]
    text_main = "#e2e8f0" if st.session_state.theme == "night" else "#1a202c"
    text_sub  = "#a0aec0" if st.session_state.theme == "night" else "#4a5568"
    
    st.markdown(f'<div style="text-align: center; font-size: 42px; font-weight: 900; color: #00b4d8; margin-top: 30px;">⚙️ 选择难度</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center; font-size: 16px; color: {text_sub}; margin-bottom: 40px;">当前病例：{case["name"]} - {case["disease"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05)); 
                    border: 1px solid rgba(34, 197, 94, 0.4); border-radius: 16px; padding: 25px; 
                    margin-bottom: 20px; height: 230px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; font-weight: 800; color: #4ade80; margin-bottom: 15px;">🌱 简单模式</div>
            <ul style="line-height: 2.2; color: {text_main}; margin-left: 20px; margin-bottom: 0; font-size: 15px;">
                <li><b style="color: #4ade80;">5</b> 个行动点</li>
                <li>初始信任值 <b style="color: #4ade80;">40</b></li>
                <li>有详细提示和引导</li>
                <li>适合初次体验游戏</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ 开始简单模式", use_container_width=True, type="primary", key="btn_easy"):
            st.session_state.game_mode = "easy"; st.session_state.screen = "game"; st.rerun()
            
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05)); 
                    border: 1px solid rgba(239, 68, 68, 0.4); border-radius: 16px; padding: 25px; 
                    margin-bottom: 20px; height: 230px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
            <div style="font-size: 24px; font-weight: 800; color: #f87171; margin-bottom: 15px;">🔥 困难模式</div>
            <ul style="line-height: 2.2; color: {text_main}; margin-left: 20px; margin-bottom: 0; font-size: 15px;">
                <li>仅 <b style="color: #f87171;">3</b> 个行动点</li>
                <li>初始信任值 <b style="color: #f87171;">20</b></li>
                <li>无提示，盲盒线索</li>
                <li>高压生死挑战</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶ 开始困难模式", use_container_width=True, key="btn_hard"):
            st.session_state.game_mode = "hard"; st.session_state.screen = "game"; st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    _, c_mid, _ = st.columns([1, 1, 1])
    with c_mid:
        if st.button("⬅️ 返回病例选择", use_container_width=True): st.session_state.screen = "case_select"; st.rerun()
    st.stop()

if st.session_state.screen == "profile":
    st.markdown('<div style="text-align:center; font-size:42px; font-weight:900; color:#00b4d8; margin:20px 0;">📕 我的错题本</div>', unsafe_allow_html=True)
    st.caption(f"👨‍⚕️ 当前玩家：**{st.session_state.player_name}**")
    records = fetch_my_records(st.session_state.player_name)
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
            mistakes = _json.loads(r["mistakes"] or "[]")
            with st.expander(f"{'✅' if r['survived'] else '💀'} {r['created_at']} | {r['disease']} | 得分 {r['final_score']}"):
                a, b = st.columns(2)
                a.write(f"**难度**：{r['mode']}\n\n**病情度**：{r['disease_progress']}\n\n**问诊/诊断/急救/沟通**：{r['score_inquiry']}/{r['score_diagnosis']}/{r['score_emergency']}/{r['score_empathy']}")
                with b:
                    if mistakes: st.warning("**失误记录：**\n" + "\n".join(f"- {m}" for m in mistakes))
                    else: st.success("本局无失误 🎉")
    if st.button("⬅️ 返回主菜单"): st.session_state.screen = "start"; st.rerun()
    st.stop()

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
        agg = df.groupby("disease").agg(场次=("id", "count"), 平均分=("final_score", "mean"), 最高分=("final_score", "max"), 存活率=("survived", "mean")).reset_index()
        agg["平均分"] = agg["平均分"].round(1)
        agg["存活率"] = (agg["存活率"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(agg, use_container_width=True, hide_index=True)
        st.subheader("📊 各病例平均分对比")
        fig2 = px.bar(agg, x="disease", y="平均分", color="平均分", color_continuous_scale="Teal", text="平均分")
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("🏅 成就解锁率")
        total = len(df)
        ach_stats = []
        for aid, a in ACHIEVEMENTS.items():
            cnt = sum(1 for x in df["achievements"] if aid in _json.loads(x or "[]"))
            ach_stats.append({"成就": f"{a['icon']} {a['name']}", "解锁率": f"{cnt/total*100:.0f}%", "解锁人数": cnt})
        st.dataframe(pd.DataFrame(ach_stats), use_container_width=True, hide_index=True)
    if st.button("⬅️ 返回主菜单"): st.session_state.screen = "start"; st.rerun()
    st.stop()

# ================= 11. 危机倒计时 =================
@st.fragment(run_every="1s")
def crisis_countdown():
    if st.session_state.get("game_over"): return
    elapsed = time.time() - st.session_state.get("crisis_start_time", time.time())
    time_left = max(0, 90 - int(elapsed))
    if time_left <= 0:
        st.session_state.disease_progress = 100
        st.session_state.game_over = True
        st.rerun()
    st.error(f"⏰ 紧急抢救倒计时：{time_left} 秒！")

# ================= 12. 游戏主逻辑 =================
if st.session_state.screen == "game":
    case = CASES_CONFIG[st.session_state.current_case]
    mode = st.session_state.game_mode

    defaults = {
        "messages": [{"role": "assistant", "content": "医生！您快看看我家孩子！"}],
        "trust_score": 40 if mode == "easy" else 20,
        "action_points": 5 if mode == "easy" else 3,
        "disease_progress": 30,
        "game_over": False, "time_period": 1, "unlocked_clues": [], "final_evaluation": None,
        "decision_made": {}, "penalty_log": [], "crisis_actions": [], "crisis_correct_count": 0,
        "final_score": 0, "final_title": "", "score_inquiry": 0, "score_diagnosis": 0, "score_emergency": 0, "score_empathy": 0,
        "diagnosis_made": None, "diagnosis_processed": False, "auscultation_mode": False, "auscultation_completed": False,
        "physical_exam_done": {}, "achievements": st.session_state.get("achievements", []),
        "first_act_clue": False, "excluded_diseases": [],
        "comm_made": False, "last_action_time": time.time(), "crisis_start_time": time.time(),
        "record_saved": False, "intro_video_played": False
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    check_achievements(case)

    SCENARIO_DATA = {
        1: {"time": "凌晨 2:00", "title": "🎬 第一幕：急诊室初遇", "mode": "free"},
        2: {"time": "凌晨 2:30", "title": "🎬 第二幕：初步判断", "mode": "decision"},
        3: {"time": "凌晨 2:45", "title": "🎬 第三幕：迷雾重重", "mode": "free"},
        4: {"time": "凌晨 3:00", "title": "🎬 第四幕：生死时速", "mode": "crisis"},
        5: {"time": "凌晨 3:15", "title": "🎬 第五幕：医患沟通", "mode": "communication"},
        6: {"time": "凌晨 3:30", "title": "🎬 第六幕：结案复盘", "mode": "end"}
    }

    st.sidebar.markdown("---")
    st.sidebar.caption(f"当前病例：{case['disease']}")
    if st.sidebar.button("🏠 返回主菜单"):
        for key in list(st.session_state.keys()):
            if key not in ["theme", "player_id", "player_name", "achievements"]: del st.session_state[key]
        st.session_state.screen = "start"; st.rerun()
    render_achievements()

    if st.session_state.time_period == 1 and not st.session_state.intro_video_played:
        st.subheader("🎬 急诊室画面")
        if os.path.exists("intro_er.mp4"): st.video("intro_er.mp4", autoplay=True, muted=True)
        else: st.info("（此处为开场视频：深夜急诊，一位母亲抱着孩子冲进抢救室…）")
        if st.button("⏭️ 跳过开场"):
            st.session_state.intro_video_played = True; st.rerun()
        st.session_state.intro_video_played = True

    st.markdown(f'<div style="display: flex; justify-content: space-around; gap: 15px; margin-bottom: 25px;"><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">❤️ 信任值</div><div style="font-size:20px; font-weight:800; color:#00b4d8;">{st.session_state.trust_score}</div></div><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">⏳ 行动点</div><div style="font-size:20px; font-weight:800; color:#f6e05e;">{st.session_state.action_points}</div></div><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">⚠️ 病情度</div><div style="font-size:20px; font-weight:800; color:#e63946;">{st.session_state.disease_progress}</div></div></div>', unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1.1, 2.5, 1.2])

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
            tp = st.session_state.time_period
            ap = st.session_state.action_points

            if tp == 1:
                if ap <= 2:
                    if st.button("▶️ 进入第二幕：初步判断", use_container_width=True, type="primary"):
                        st.session_state.time_period = 2; st.session_state.disease_progress += 10; st.rerun()
                else: st.caption(f"💡 建议再问诊 {ap - 2} 次（剩余行动点 {ap}）")

            if tp == 3:
                if st.button("🚨 突发事件！进入第四幕", use_container_width=True, type="primary"):
                    st.session_state.time_period = 4; st.session_state.action_points = 3
                    st.session_state.disease_progress = max(70, st.session_state.disease_progress + 15)
                    st.session_state.crisis_start_time = time.time()
                    st.session_state.messages.append({"role": "assistant", "content": "（患儿突然剧烈哭闹，病情急剧加重）医生！救救他！"})
                    st.rerun()

            if tp == 4 and st.session_state.crisis_correct_count >= 2:
                if st.button("✅ 进入第五幕", use_container_width=True, type="primary"):
                    st.session_state.score_emergency += 20; st.session_state.time_period = 5
                    st.session_state.action_points = 5
                    st.session_state.messages.append({"role": "assistant", "content": "（经过处理，患儿生命体征平稳）"})
                    st.rerun()

            if tp == 5 and not st.session_state.comm_made:
                st.warning("⚠️ 交代病情：")
                if st.button("A. 赶紧办住院", key="c_a"):
                    st.session_state.comm_made = True; st.session_state.trust_score -= 10; st.rerun()
                if st.button("B. 病情很重，可能窒息/危险", key="c_b"):
                    st.session_state.comm_made = True; st.session_state.trust_score -= 5; st.rerun()
                if st.button("C. 您别自责，送来及时，我们会尽力", key="c_c"):
                    st.session_state.comm_made = True; st.session_state.trust_score += 15; st.session_state.score_empathy += 15; st.rerun()
            if tp == 5 and st.session_state.comm_made:
                if st.button("▶️ 进入复盘", use_container_width=True): st.session_state.time_period = 6; st.rerun()

        st.divider()
        if st.session_state.time_period >= 5 and not st.session_state.game_over:
            if st.button("📝 提交诊断，结束", use_container_width=True): st.session_state.game_over = True; st.rerun()
        if st.button("🔄 重新开始", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "screen", "current_case", "game_mode", "player_id", "player_name", "achievements"]: del st.session_state[key]
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
                else: st.text(f"🔒 未知线索")
        if st.session_state.penalty_log:
            st.divider(); st.caption("📝 失误记录")
            for log in st.session_state.penalty_log: st.warning(log)

    with col_center:
        if not st.session_state.game_over:
            st.title(SCENARIO_DATA[st.session_state.time_period]["title"])
            tp_now = st.session_state.time_period
            
            if tp_now == 1:
                st.markdown(f"""
                <div style="background: rgba(0, 180, 216, 0.1); border-left: 4px solid #00b4d8; 
                            padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <div style="font-size: 15px; font-weight: bold; color: #00b4d8; margin-bottom: 5px;">🚑 前情提要</div>
                    <div style="font-size: 14px; color: #e2e8f0; line-height: 1.6;">
                        <b>患儿：</b>{case['patient']} <br>
                        <b>场景：</b>{case['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.caption("💡 任务：安抚情绪，通过问诊了解发病时间和特征。")
            elif tp_now == 2: st.info("💡 任务：综合已有线索，做出初步诊断。")
            elif tp_now == 3: st.info("💡 任务：继续收集线索，随时准备迎接突发状况。")
        else:
            st.title("🏥 结案复盘")

        if st.session_state.time_period == 2 and not st.session_state.game_over:
            st.markdown("#### 🩺 初步诊断")
            if not st.session_state.diagnosis_processed:
                st.write("根据目前的线索，你认为最可能的诊断是：")
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
                            if st.session_state.disease_progress >= 100: st.session_state.game_over = True
                        st.rerun()
            else:
                st.success("✅ 初步诊断已记录。")
                if st.button("▶️ 进入第三幕：迷雾重重", use_container_width=True, type="primary"):
                    st.session_state.time_period = 3; st.session_state.action_points = 5; st.session_state.disease_progress += 10; st.rerun()
            with st.container(height=300):
                for msg in st.session_state.messages: render_message(msg["role"], msg["content"])

        else:
            if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
                with st.expander("🩺 体格检查工具箱（消耗行动点）", expanded=False):
                    tab1, tab2, tab3 = st.tabs(["👁️ 视诊", "👂 听诊", "🖐️ 触诊/叩诊"])
                    with tab1:
                        if st.button("观察呼吸/查体"):
                            if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("vis"):
                                st.session_state.action_points -= 1
                                st.session_state.physical_exam_done["vis"] = True
                                if len(case["clues"]) > 0:
                                    st.session_state.unlocked_clues.append(case["clues"][-1])
                                st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                                st.session_state.messages.append({"role": "assistant", "content": "（你观察到患儿明显的阳性体征。）"})
                                st.rerun()
                    with tab2:
                        if st.button("进行听诊"):
                            if st.session_state.action_points > 0 and not st.session_state.auscultation_completed:
                                st.session_state.action_points -= 1
                                st.session_state.auscultation_mode = True; st.rerun()

            if st.session_state.auscultation_mode:
                st.warning("🩺 听诊音频：")
                if os.path.exists(case["audio"]): st.audio(case["audio"], format="audio/mp3")
                st.write("请判断：")
                if st.button("异常呼吸音/杂音"):
                    st.session_state.unlocked_clues.append(case["clues"][0])
                    st.session_state.score_diagnosis += 10
                    st.session_state.auscultation_mode = False
                    st.session_state.auscultation_completed = True; st.rerun()
                if st.button("正常呼吸音"):
                    st.session_state.auscultation_mode = False
                    st.session_state.auscultation_completed = True; st.rerun()
            else:
                with st.container(height=400):
                    for msg in st.session_state.messages: render_message(msg["role"], msg["content"])

                if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "crisis":
                    crisis_countdown()
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
                    else:
                        if st.session_state.crisis_correct_count < 2:
                            st.error("🚨 行动点耗尽！患儿情况危急，请呼叫支援！")
                            if st.button("📞 呼叫上级医师支援（恢复2点行动点，病情+15）", use_container_width=True):
                                st.session_state.action_points = 2
                                st.session_state.disease_progress = min(100, st.session_state.disease_progress + 15)
                                st.session_state.score_emergency -= 5; st.rerun()

                elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
                    if prompt := st.chat_input("输入问诊或检查..."):
                        if st.session_state.action_points <= 0: st.warning("行动点已用完，请进入下一幕。")
                        else:
                            st.session_state.action_points -= 1
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            hit_clues = match_clues(prompt, case)
                            new_hits = [c for c in hit_clues if c not in st.session_state.unlocked_clues]
                            for clue in new_hits:
                                st.session_state.unlocked_clues.append(clue)
                                st.session_state.score_inquiry += 8
                                st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                            if new_hits: st.session_state.first_act_clue = True
                            
                            with st.spinner("妈妈正在回答…"):
                                reply = call_mom_llm(st.session_state.current_case, case, st.session_state.messages, hit_clues)
                            if not reply: reply = pick_fallback(hit_clues)
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                            st.rerun()

    if st.session_state.game_over:
        st.divider(); st.header("🩺 带教老师复盘")
        if st.session_state.final_score == 0:
            total = (st.session_state.score_inquiry + st.session_state.score_diagnosis + st.session_state.score_emergency + st.session_state.score_empathy)
            st.session_state.final_score = total
            if total >= 80: st.session_state.final_title = "🏆 儿科急诊之光"
            elif total >= 60: st.session_state.final_title = "🌟 有潜力的住院医"
            else: st.session_state.final_title = "📚 还需回炉重造"

        if not st.session_state.record_saved:
            check_achievements(case)
            save_record({
                "player_id": st.session_state.player_id, "player_name": st.session_state.player_name,
                "case_id": st.session_state.current_case, "case_name": case["name"], "disease": case["disease"],
                "mode": st.session_state.game_mode, "final_score": st.session_state.final_score,
                "score_inquiry": st.session_state.score_inquiry, "score_diagnosis": st.session_state.score_diagnosis,
                "score_emergency": st.session_state.score_emergency, "score_empathy": st.session_state.score_empathy,
                "disease_progress": st.session_state.disease_progress, "survived": 1 if st.session_state.disease_progress < 100 else 0,
                "mistakes": _json.dumps(st.session_state.penalty_log, ensure_ascii=False),
                "achievements": _json.dumps(st.session_state.achievements, ensure_ascii=False),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.session_state.record_saved = True

        st.info(f"### {st.session_state.final_title}\n最终得分：{st.session_state.final_score} / 100")
        st.subheader("📊 个人能力雷达图")
        fig = go.Figure(go.Scatterpolar(
            r=[min(100, st.session_state.score_inquiry * 3), min(100, st.session_state.score_diagnosis * 2), min(100, st.session_state.score_emergency * 2), min(100, st.session_state.score_empathy * 2)],
            theta=['问诊', '诊断', '急救', '沟通'], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("🏅 本局成就")
        render_achievements(container=st)

        if st.button("🏠 返回主菜单"):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "player_id", "player_name", "achievements"]: del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()
