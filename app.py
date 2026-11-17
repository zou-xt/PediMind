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
from collections import Counter

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

# ================= 2.5 AI 对话层（表演层）与线索判定层 =================
CLUE_SYNONYMS = {
    "犬吠样咳嗽":     ["小狗叫", "狗叫", "海豹叫", "空空声", "像狗", "犬吠", "croup"],
    "夜间加重":       ["半夜", "夜里", "晚上更", "夜间", "睡着后", "后半夜"],
    "吸气性喉鸣":     ["吸气有声", "吸气响", "喉咙响", "吸气鸣", "拉风箱", "吸气的时候有声音"],
    "白天感冒史":     ["白天感冒", "之前感冒", "流鼻涕", "有点感冒", "咳嗽几天", "着凉"],
    "三凹征":         ["胸口凹", "锁骨凹", "肋骨凹", "吸气凹", "胸骨上窝", "凹陷", "凹进去"],
    "进食呛咳史":     ["吃花生", "吃东西呛", "吃坚果", "边吃边咳", "呛到", "吃东西时咳"],
    "剧烈呛咳":       ["咳得厉害", "剧烈咳嗽", "咳得弓起来", "咳不停", "咳得停不下来"],
    "呼吸音不对称":   ["两边呼吸不一样", "一边响一边不响", "呼吸不对称", "两边声音不同"],
    "面色发绀":       ["脸发紫", "嘴唇紫", "脸憋紫", "脸色青紫", "脸都紫了"],
    "既往哮喘史":     ["以前喘过", "有哮喘", "以前也喘", "小时候就喘", "老毛病"],
    "过敏史":         ["过敏", "对花粉", "湿疹", "花粉过敏"],
    "呼气性呼吸困难": ["呼气费力", "呼气长", "呼气困难", "出气费劲", "出气比进气长"],
    "双肺哮鸣音":     ["喘鸣", "呼呼响", "肺里有声音", "哨音", "吹哨子", "吱吱响"],
    "高热":           ["发烧", "高烧", "39度", "40度", "体温很高", "烧得厉害"],
    "流涎":           ["流口水", "一直流口水", "口水多", "口水流不停"],
    "吞咽困难":       ["咽不下", "不敢咽", "吞东西疼", "不肯吃东西", "咽东西费劲"],
    "拒绝平卧":       ["一躺下就", "不肯躺", "躺下就喘", "坐起来", "只能坐着", "不愿平躺"],
    "无犬吠样咳嗽":   ["不是狗叫", "不像狗叫", "没有狗叫", "不是犬吠", "咳嗽声音不一样"],
}

BANNED_TERMS = [
    "急性", "喉炎", "会厌", "哮喘", "异物", "支气管", "诊断", "确诊",
    "三凹征", "哮鸣", "喉鸣", "发绀", "呼吸困难", "Stridor", "Wheezing",
    "糖皮质激素", "肾上腺素", "雾化", "地塞米松", "海姆立克", "插管", "气管切开",
]

MOM_FACTS = {
    "laryngitis": [
        "他咳起来像小狗叫，空空的，我从来没听过这种声音",
        "半夜突然加重，之前白天有点流鼻涕、着凉",
        "吸气的时候喉咙里有响声，胸口还会凹进去一块",
        "孩子很烦躁，坐起来好像好一点",
    ],
    "foreign_body": [
        "刚才在吃花生，突然就咳得整个人弓起来了",
        "咳得停不下来，脸都憋紫了",
        "一边呼吸声大一边小，听着不对劲",
        "孩子很害怕，一直哭",
    ],
    "asthma": [
        "她以前也这样喘过好几次，小时候就有这毛病",
        "对花粉过敏，小时候长过湿疹",
        "这次呼气很费力，出气比进气长",
        "肺里呼呼响，像吹哨子一样",
    ],
    "epiglottitis": [
        "发高烧，39度多，烫得吓人",
        "一直流口水，咽都咽不下去，不肯吃东西",
        "一躺下就喘不上气，非要坐着",
        "咳嗽不是那种小狗叫的声音",
    ],
}

FALLBACK_REPLIES = {
    "犬吠样咳嗽":   "他咳起来像小狗叫，我从来没听过这种声音…",
    "三凹征":       "他吸气的时候，胸口会凹进去一块，看着吓人…",
    "进食呛咳史":   "刚才在吃花生，突然就咳得整个人弓起来了…",
    "剧烈呛咳":     "他咳得停不下来，整个人都弓起来了…",
    "面色发绀":     "他脸都憋紫了，医生你快看看啊…",
    "流涎":         "他一直流口水，咽都咽不下去…",
    "拒绝平卧":     "他一躺下就喘不上气，非要坐着…",
    "高热":         "他烧到39度多，烫得吓人…",
    "既往哮喘史":   "他以前也这样喘过好几次…",
    "双肺哮鸣音":   "他肺里呼呼响，像吹哨子一样…",
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
    if not isinstance(reply, str):
        return None
    reply = reply.strip()
    if not reply or len(reply) > 80:
        return None
    if any(t in reply for t in BANNED_TERMS):
        return None
    return reply

def pick_fallback(hit_clues: list) -> str:
    for c in hit_clues:
        if c in FALLBACK_REPLIES:
            return FALLBACK_REPLIES[c]
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
            out.append({"role": "user" if m.get("role") == "user" else "assistant",
                        "content": c})
    return out

def call_mom_llm(case_id: str, case: dict, history: list, hit_clues: list):
    if not _ai_available():
        return None
    system = build_mom_system(case_id, case)
    msgs = [{"role": "system", "content": system}] + _clean_history(history[-6:])
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=msgs,
                temperature=0.8,
                max_tokens=120,
                timeout=6,
                response_format={"type": "json_object"},
            )
            data = _json.loads(resp.choices[0].message.content)
            reply = sanitize_reply(data.get("reply"))
            if reply:
                return reply
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
    if any(m.get("role") == "user" for m in ss.messages):
        unlock_achievement("first_inquiry")
    if len(ss.unlocked_clues) >= len(case["clues"]):
        unlock_achievement("all_clues")
    if ss.diagnosis_made and case["diagnosis"].get(ss.diagnosis_made) == case["disease"]:
        unlock_achievement("correct_dx")
    if ss.crisis_correct_count >= 2:
        unlock_achievement("lifesaver")
    if ss.score_empathy > 0:
        unlock_achievement("empathy")
    if ss.final_score >= 80:
        unlock_achievement("perfect")

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

# ================= 4. 数据持久化层 (SQLite) =================
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

def fetch_my_records(player_name: str):
    with db() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM records WHERE player_name=? ORDER BY id DESC",
                         (player_name,)).fetchall()
    return [dict(r) for r in rows]

def fetch_all_records():
    with db() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]

# ================= 5. 头像 & 消息气泡 =================
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

# ================= 6. 优化版心电监护仪 =================
def render_monitor(disease_progress: int):
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
      body {{ background:#02060a; overflow:hidden; }}
      .monitor {{ position:relative; width:100%; height:180px; background:#02060a;
                 border-radius:10px; border:1px solid #0a3a2a; overflow:hidden; }}
      canvas {{ display:block; width:100%; height:100%; }}
      
      /* 背景加深，防止波形穿字 */
      .vitals {{ position:absolute; top:8px; right:12px; text-align:right;
                font-family:'Courier New',monospace;
                background: rgba(0, 0, 0, 0.65); 
                padding: 8px 12px; border-radius: 8px; 
                backdrop-filter: blur(4px); z-index: 10; }}
      .vital-item {{ margin-bottom:4px; line-height:1.1; text-shadow:0 0 6px currentColor; }}
      .vital-label {{ font-size:10px; opacity:0.85; letter-spacing:1px; }}
      .vital-value {{ font-size:24px; font-weight:bold; }}
      .vital-unit  {{ font-size:10px; opacity:0.65; }}
      
      .c-green {{ color:#22ff88; }} .c-yellow {{ color:#ffcc00; }} .c-red {{ color:#ff3b3b; }}
      .blink {{ animation:blink 0.6s infinite; }}
      @keyframes blink {{ 50% {{ opacity:0.25; }} }}
    </style></head><body>
      <div class="monitor">
        <canvas id="ecg"></canvas>
        <div class="vitals">
          <div class="vital-item"><div class="vital-label">HR</div><div id="hr" class="vital-value">--</div><div class="vital-unit">bpm</div></div>
          <div class="vital-item"><div class="vital-label">SpO₂</div><div id="spo2" class="vital-value">--</div><div class="vital-unit">%</div></div>
          <div class="vital-item"><div class="vital-label">RR</div><div id="rr" class="vital-value">--</div><div class="vital-unit">/min</div></div>
        </div>
      </div>
    <script>
      const HR={hr}, SPO2={spo2}, RR={rr}, STATUS="{status}";
      const color = (STATUS==="critical"||STATUS==="danger") ? "#ff3b3b"
                  : STATUS==="warn" ? "#ffcc00" : "#22ff88";
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

      // 真实感ECG波形
      function ecgSample(p) {{
        let v = 0;
        v += 0.12 * Math.exp(-Math.pow((p-0.15)/0.015, 2)); 
        v -= 0.15 * Math.exp(-Math.pow((p-0.24)/0.005, 2)); 
        v += 1.15 * Math.exp(-Math.pow((p-0.25)/0.006, 2)); 
        v -= 0.25 * Math.exp(-Math.pow((p-0.265)/0.008, 2)); 
        v += 0.25 * Math.exp(-Math.pow((p-0.4)/0.03, 2));   
        return v;
      }}

      const BUFFER = 400;
      const data = new Array(BUFFER).fill(0);
      let phase = 0;
      const cycleDur = 60 / HR;
      let lastTime = performance.now();

      function drawGrid() {{
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = "rgba(0,255,128,0.06)";
        for (let x=0; x<W; x+=8) {{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }}
        for (let y=0; y<H; y+=8) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }}
        ctx.strokeStyle = "rgba(0,255,128,0.12)";
        for (let x=0; x<W; x+=40) {{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }}
        for (let y=0; y<H; y+=40) {{ ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }}
      }}

      function drawWave() {{
        const step = W / BUFFER, midY = H * 0.62, amp = H * 0.42;
        ctx.beginPath();
        ctx.strokeStyle = color; 
        ctx.lineWidth = 1.5;
        ctx.lineJoin = 'round';
        for (let i=0; i<BUFFER; i++) {{
          const x = i*step, y = midY - data[i]*amp;
          i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
        }}
        ctx.stroke(); 
      }}

      function loop(now) {{
        let dt = (now - lastTime) / 1000;
        lastTime = now;
        if (dt > 0.1) dt = 0.1; 

        const sampleRate = 60; 
        const perFrame = Math.max(1, Math.floor(dt * sampleRate));
        const dPhase = (dt / cycleDur);

        for (let k=0; k<perFrame; k++) {{
          phase = (phase + dPhase / perFrame) % 1;
          let v = ecgSample(phase);
          if (STATUS==="danger"||STATUS==="critical") {{
              v += (Math.random()-0.5)*0.05;
          }}
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
        hrEl.textContent = jit(HR,4); spEl.textContent = jit(SPO2,1); rrEl.textContent = jit(RR,1);
        [hrEl,spEl,rrEl].forEach(e=>e.className=cls);
      }}
      updateVitals(); setInterval(updateVitals, 1200);
    </script></body></html>
    """
    components.html(html_code, height=190, scrolling=False)

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
        
        /* 修复提示框内部文字对比度 */
        div[data-testid="stAlert"] * {
            color: #1a202c !important;
        }
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

# 屏幕 1：开始界面
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
        st.text_input("👨‍⚕️ 你的昵称（用于记录错题本，输入相同昵称可查看历史）",
                      key="player_name", max_chars=12)
        if st.button("🚀 开始游戏", use_container_width=True, type="primary"):
            st.session_state.screen = "case_select"; st.rerun()
        c1, c2 = st.columns(2)
        if c1.button("📕 我的错题本", use_container_width=True):
            st.session_state.screen = "profile"; st.rerun()
        if c2.button("📊 全服统计", use_container_width=True):
            st.session_state.screen = "stats"; st.rerun()
    st.stop()

# 屏幕 2：病例选择
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
                st.session_state.current_case = case_id
                st.session_state.screen = "difficulty_select"; st.rerun()
    if st.button("⬅️ 返回主菜单"): st.session_state.screen = "start"; st.rerun()
    st.stop()

# 屏幕 3：难度选择（优化版界面）
if st.session_state.screen == "difficulty_select":
    case = CASES_CONFIG[st.session_state.current_case]
    
    # 根据主题动态调整文字颜色
    text_main = "#e2e8f0" if st.session_state.theme == "night" else "#1a202c"
    text_sub  = "#a0aec0" if st.session_state.theme == "night" else "#4a5568"
    
    st.markdown(f'<div style="text-align: center; font-size: 42px; font-weight: 900; color: #00b4d8; margin-top: 30px;">⚙️ 选择难度</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: center; font-size: 16px; color: {text_sub}; margin-bottom: 40px;">当前病例：{case["name"]} - {case["disease"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # 简单模式卡片
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
            st.session_state.game_mode = "easy"
            st.session_state.screen = "game"
            st.rerun()
            
    with col2:
        # 困难模式卡片
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
            st.session_state.game_mode = "hard"
            st.session_state.screen = "game"
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    _, c_mid, _ = st.columns([1, 1, 1])
    with c_mid:
        if st.button("⬅️ 返回病例选择", use_container_width=True):
            st.session_state.screen = "case_select"
            st.rerun()
    st.stop()

# ================= 9. 错题本 =================
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
        fig = px.line(df.iloc[::-1], x="created_at", y="final_score",
                      markers=True, color="case_name")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🗂️ 历史记录")
        for r in records:
            mistakes = _json.loads(r["mistakes"] or "[]")
            with st.expander(f"{'✅' if r['survived'] else '💀'} {r['created_at']} | {r['disease']} | 得分 {r['final_score']}"):
                a, b = st.columns(2)
                a.write(f"**难度**：{r['mode']}\n\n**病情度**：{r['disease_progress']}\n\n"
                        f"**问诊/诊断/急救/沟通**：{r['score_inquiry']}/{r['score_diagnosis']}/{r['score_emergency']}/{r['score_empathy']}")
                with b:
                    if mistakes:
                        st.warning("**失误记录：**\n" + "\n".join(f"- {m}" for m in mistakes))
                    else:
                        st.success("本局无失误 🎉")

    if st.button("⬅️ 返回主菜单"):
        st.session_state.screen = "start"; st.rerun()
    st.stop()

# ================= 10. 全服统计 =================
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
            场次=("id", "count"),
            平均分=("final_score", "mean"),
            最高分=("final_score", "max"),
            存活率=("survived", "mean")).reset_index()
        agg["平均分"] = agg["平均分"].round(1)
        agg["存活率"] = (agg["存活率"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(agg, use_container_width=True, hide_index=True)

        st.subheader("📊 各病例平均分对比")
        fig2 = px.bar(agg, x="disease", y="平均分", color="平均分",
                      color_continuous_scale="Teal", text="平均分")
        fig2.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🏅 成就解锁率")
        total = len(df)
        ach_stats = []
        for aid, a in ACHIEVEMENTS.items():
            cnt = sum(1 for x in df["achievements"] if aid in _json.loads(x or "[]"))
            ach_stats.append({"成就": f"{a['icon']} {a['name']}",
                              "解锁率": f"{cnt/total*100:.0f}%", "解锁人数": cnt})
        st.dataframe(pd.DataFrame(ach_stats), use_container_width=True, hide_index=True)

    if st.button("⬅️ 返回主菜单"):
        st.session_state.screen = "start"; st.rerun()
    st.stop()

# ================= 11. 危机倒计时（实时刷新）=================
@st.fragment(run_every="1s")
def crisis_countdown():
    if st.session_state.get("game_over"):
        return
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
        "messages": [{"role": "assistant", "content": f"医生！您快看看我家孩子！{case['description']}"}],
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
            if key not in ["theme", "player_id", "player_name", "achievements"]:
                del st.session_state[key]
        st.session_state.screen = "start"; st.rerun()
    render_achievements()

    if st.session_state.time_period == 1 and not st.session_state.intro_video_played:
        st.subheader("🎬 急诊室画面")
        if os.path.exists("intro_er.mp4"):
            st.video("intro_er.mp4", autoplay=True, muted=True)
        else:
            st.info("（此处为开场视频：深夜急诊，一位母亲抱着孩子冲进抢救室…）")
        if st.button("⏭️ 跳过开场"):
            st.session_state.intro_video_played = True; st.rerun()
        st.session_state.intro_video_played = True

    st.markdown(f'<div style="display: flex; justify-content: space-around; gap: 15px; margin-bottom: 25px;"><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">❤️ 信任值</div><div style="font-size:20px; font-weight:800; color:#00b4d8;">{st.session_state.trust_score}</div></div><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">⏳ 行动点</div><div style="font-size:20px; font-weight:800; color:#f6e05e;">{st.session_state.action_points}</div></div><div class="status-badge"><div style="font-size:12px; color:#a0aec0;">⚠️ 病情度</div><div style="font-size:20px; font-weight:800; color:#e63946;">{st.session_state.disease_progress}</div></div></div>', unsafe_allow_html=True)

    col_left, col_center, col_right = st.columns([1, 2.5, 1.2])

    # ---------------- 左侧 ----------------
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

            # 第一幕 → 第二幕
            if tp == 1:
                if ap <= 2:
                    if st.button("▶️ 进入第二幕：初步判断", use_container_width=True, type="primary"):
                        st.session_state.time_period = 2
                        st.session_state.disease_progress += 10
                        st.rerun()
                else:
                    st.caption(f"💡 建议再问诊 {ap - 2} 次（剩余行动点 {ap}）")

            # 第三幕 → 第四幕
            if tp == 3:
                if st.button("🚨 突发事件！进入第四幕", use_container_width=True, type="primary"):
                    st.session_state.time_period = 4
                    st.session_state.action_points = 3
                    st.session_state.disease_progress = max(70, st.session_state.disease_progress + 15)
                    st.session_state.crisis_start_time = time.time()
                    st.session_state.messages.append({"role": "assistant", "content": "（患儿突然剧烈哭闹，呼吸困难急剧加重）医生！救救她！"})
                    st.rerun()

            # 第四幕 → 第五幕
            if tp == 4 and st.session_state.crisis_correct_count >= 2:
                if st.button("✅ 进入第五幕", use_container_width=True, type="primary"):
                    st.session_state.score_emergency += 20
                    st.session_state.time_period = 5
                    st.session_state.action_points = 5
                    st.session_state.messages.append({"role": "assistant", "content": "（经过处理，患儿呼吸平稳）"})
                    st.rerun()

            # 第五幕：沟通
            if tp == 5 and not st.session_state.comm_made:
                st.warning("⚠️ 交代病情：")
                if st.button("A. 赶紧办住院", key="c_a"):
                    st.session_state.comm_made = True; st.session_state.trust_score -= 10; st.rerun()
                if st.button("B. 病情很重，可能窒息", key="c_b"):
                    st.session_state.comm_made = True; st.session_state.trust_score -= 5; st.rerun()
                if st.button("C. 您别自责，送来及时，我们会尽力", key="c_c"):
                    st.session_state.comm_made = True; st.session_state.trust_score += 15
                    st.session_state.score_empathy += 15; st.rerun()
            if tp == 5 and st.session_state.comm_made:
                if st.button("▶️ 进入复盘", use_container_width=True):
                    st.session_state.time_period = 6; st.rerun()

        st.divider()
        if st.session_state.time_period >= 5 and not st.session_state.game_over:
            if st.button("📝 提交诊断，结束", use_container_width=True):
                st.session_state.game_over = True; st.rerun()
        if st.button("🔄 重新开始", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "screen", "current_case", "game_mode",
                               "player_id", "player_name", "achievements"]:
                    del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()

    # ---------------- 右侧：线索夹 ----------------
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

    # ---------------- 中间：对话与核心交互 ----------------
    with col_center:
        if not st.session_state.game_over:
            st.title(SCENARIO_DATA[st.session_state.time_period]["title"])
            tp_now = st.session_state.time_period
            if tp_now == 1:
                st.info("任务：安抚情绪，了解发病时间和特征。")
            elif tp_now == 2:
                st.info("任务：综合已有线索，做出初步诊断。")
            elif tp_now == 3:
                st.info("任务：继续收集线索，随时准备迎接突发状况。")
        else:
            st.title("🏥 结案复盘")

        # ========== 第二幕专用：初步诊断 ==========
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
                            if st.session_state.disease_progress >= 100:
                                st.session_state.game_over = True
                        st.rerun()
            else:
                st.success("✅ 初步诊断已记录。")
                if st.button("▶️ 进入第三幕：迷雾重重", use_container_width=True, type="primary"):
                    st.session_state.time_period = 3
                    st.session_state.action_points = 5
                    st.session_state.disease_progress += 10
                    st.rerun()
            with st.container(height=300):
                for msg in st.session_state.messages:
                    render_message(msg["role"], msg["content"])

        # ========== 其他幕：原有逻辑 ==========
        else:
            # 体格检查（仅 free 模式）
            if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
                with st.expander("🩺 体格检查工具箱（消耗行动点）", expanded=False):
                    tab1, tab2, tab3 = st.tabs(["👁️ 视诊", "👂 听诊", "🖐️ 触诊/叩诊"])
                    with tab1:
                        if st.button("观察呼吸/查体"):
                            if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("vis"):
                                st.session_state.action_points -= 1
                                st.session_state.physical_exam_done["vis"] = True
                                st.session_state.unlocked_clues.append("三凹征" if "三凹征" in case["clues"] else case["clues"][0])
                                st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                                st.session_state.messages.append({"role": "assistant", "content": "（你观察到患儿呼吸急促，相关体征阳性。）"})
                                st.rerun()
                    with tab2:
                        if st.button("进行听诊"):
                            if st.session_state.action_points > 0 and not st.session_state.auscultation_completed:
                                st.session_state.action_points -= 1
                                st.session_state.auscultation_mode = True; st.rerun()

            # 听诊模式
            if st.session_state.auscultation_mode:
                st.warning("🩺 听诊音频：")
                if os.path.exists(case["audio"]): st.audio(case["audio"], format="audio/mp3")
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

                # 危机模式
                if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "crisis":
                    crisis_countdown()
                    if st.session_state.action_points > 0:
                        for opt_key, opt_val in case["crisis"].items():
                            if st.button(opt_val["label"], key=f"crisis_{opt_key}", use_container_width=True):
                                st.session_state.action_points -= 1
                                st.session_state.messages.append({"role": "user", "content": opt_val["label"]})
                                st.session_state.messages.append({"role": "assistant", "content": opt_val["reply"]})
                                st.session_state.disease_progress += opt_val["disease"]
                                if opt_val["type"] == "correct":
                                    st.session_state.crisis_correct_count += 1
                                else:
                                    st.session_state.penalty_log.append(opt_val["label"])
                                if st.session_state.disease_progress >= 100:
                                    st.session_state.game_over = True
                                st.rerun()
                    else:
                        if st.session_state.crisis_correct_count < 2:
                            st.error("🚨 行动点耗尽！患儿情况危急，请呼叫支援！")
                            if st.button("📞 呼叫上级医师支援（恢复2点行动点，病情+15）", use_container_width=True):
                                st.session_state.action_points = 2
                                st.session_state.disease_progress = min(100, st.session_state.disease_progress + 15)
                                st.session_state.score_emergency -= 5
                                st.rerun()

                # 自由问诊
                elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
                    if prompt := st.chat_input("输入问诊或检查..."):
                        if st.session_state.action_points <= 0:
                            st.warning("行动点已用完，请进入下一幕。")
                        else:
                            st.session_state.action_points -= 1
                            st.session_state.messages.append({"role": "user", "content": prompt})

                            # 判定层（确定性）
                            hit_clues = match_clues(prompt, case)
                            new_hits = [c for c in hit_clues if c not in st.session_state.unlocked_clues]
                            for clue in new_hits:
                                st.session_state.unlocked_clues.append(clue)
                                st.session_state.score_inquiry += 8
                                st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                            if new_hits:
                                st.session_state.first_act_clue = True

                            # 表演层（LLM 受约束 + 兜底）
                            with st.spinner("妈妈正在回答…"):
                                reply = call_mom_llm(
                                    st.session_state.current_case, case,
                                    st.session_state.messages, hit_clues
                                )
                            if not reply:
                                reply = pick_fallback(hit_clues)

                            hint = f"（提示：已获取线索 {len(st.session_state.unlocked_clues)}/{len(case['clues'])}）"
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"{reply}\n\n{hint}"}
                            )
                            st.rerun()

    # ---------------- 结算 ----------------
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
            check_achievements(case)
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
                "mistakes": _json.dumps(st.session_state.penalty_log, ensure_ascii=False),
                "achievements": _json.dumps(st.session_state.achievements, ensure_ascii=False),
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

        st.subheader("🏅 本局成就")
        render_achievements(container=st)

        if st.button("🏠 返回主菜单"):
            for key in list(st.session_state.keys()):
                if key not in ["theme", "player_id", "player_name", "achievements"]:
                    del st.session_state[key]
            st.session_state.screen = "start"; st.rerun()
