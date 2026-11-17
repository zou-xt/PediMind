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
# ================= 3. 成就系统【通用 + 分病例专属成就】 =================
# 通用成就（所有病例都可以解锁）
COMMON_ACHIEVEMENTS = {
    "common_first_inquiry": {"icon": "🩺", "name": "初次问诊", "desc": "迈出问诊第一步"},
    "common_all_clues":     {"icon": "🔍", "name": "明察秋毫", "desc": "集齐本局全部线索"},
    "common_correct_dx":    {"icon": "🎯", "name": "一针见血", "desc": "初步诊断完全正确"},
    "common_empathy":       {"icon": "💬", "name": "医者仁心", "desc": "使用共情话术交代病情"},
    "common_perfect":       {"icon": "🏆", "name": "急诊之光", "desc": "本局得分≥80，未呼叫上级支援"},
}

# 每个病例独有的成就，key和CASES_CONFIG一一对应
CASE_SPECIAL_ACHIEVEMENTS = {
    "laryngitis": {
        "laryngitis_calm_larynx": {"icon": "🌬️", "name": "冷静控喉", "desc": "急性喉炎：保持患儿安静，不触发哭闹加重水肿"},
        "laryngitis_no_laryngoscope": {"icon": "🛡️", "name": "规避喉痉挛", "desc": "急性喉炎：没有使用压舌板强行检查咽喉，避免致命喉痉挛"},
    },
    "foreign_body": {
        "foreign_body_heim": {"icon": "⚡", "name": "生死海姆立克", "desc": "气道异物：正确实施海姆立克/拍背压胸急救"},
        "foreign_body_no_dig": {"icon": "🚫", "name": "拒绝盲目抠挖", "desc": "气道异物：没有手指盲目抠挖咽部，防止异物推深"},
    },
    "asthma": {
        "asthma_relief": {"icon": "💨", "name": "平喘先锋", "desc": "支气管哮喘：正确使用支气管舒张剂急救"},
        "asthma_no_abx": {"icon": "✅", "name": "理性不用抗生素", "desc": "支气管哮喘：没有滥用抗生素治疗哮喘"},
    },
    "epiglottitis": {
        "epiglottis_protect": {"icon": "🛡️", "name": "会厌守护神", "desc": "急性会厌炎：全程没有强行压舌板查体"},
        "epiglottis_prepare_airway": {"icon": "🩻", "name": "气道预备", "desc": "急性会厌炎：选择正确处置，提前做好气道抢救准备"},
    }
}

# 工具：合并所有成就，给展览馆用
def get_all_achievement_dict():
    all_ach = dict(COMMON_ACHIEVEMENTS)
    for case_id, ach_dict in CASE_SPECIAL_ACHIEVEMENTS.items():
        all_ach.update(ach_dict)
    return all_ach

# 解锁成就函数（自动去重，不会重复解锁）
def unlock_achievement(aid):
    if "achievements" not in st.session_state:
        st.session_state.achievements = []
    if aid not in st.session_state.achievements:
        st.session_state.achievements.append(aid)

# 结算时检查、判定本局所有成就
def check_achievements(case_id, case):
    ss = st.session_state
    if "achievements" not in ss:
        ss.achievements = []

    # -------- 通用成就判定 --------
    if any(m.get("role") == "user" for m in ss.messages):
        unlock_achievement("common_first_inquiry")
    if len(ss.unlocked_clues) >= len(case["clues"]):
        unlock_achievement("common_all_clues")
    if ss.diagnosis_made and case["diagnosis"].get(ss.diagnosis_made) == case["disease"]:
        unlock_achievement("common_correct_dx")
    if ss.score_empathy > 0:
        unlock_achievement("common_empathy")
    if ss.final_score >= 80 and not ss.get("call_support_used", False):
        unlock_achievement("common_perfect")

    # -------- 病例专属成就判定 --------
    if case_id == "laryngitis":
        harm_opt_label = "强行按压患儿，用压舌板检查咽喉部"
        if harm_opt_label not in ss.penalty_log:
            unlock_achievement("laryngitis_no_laryngoscope")
        if ss.crisis_correct_count >= 1:
            unlock_achievement("laryngitis_calm_larynx")

    elif case_id == "foreign_body":
        correct_label = "立即施行海姆立克急救法/拍背压胸"
        user_contents = [m.get("content", "") for m in ss.messages if m.get("role")=="user"]
        if correct_label in user_contents:
            unlock_achievement("foreign_body_heim")
        harm_dig = "用手指盲目抠挖咽喉"
        if harm_dig not in ss.penalty_log:
            unlock_achievement("foreign_body_no_dig")

    elif case_id == "asthma":
        correct_ast = "吸氧，雾化吸入支气管舒张剂(沙丁胺醇)"
        user_contents = [m.get("content", "") for m in ss.messages if m.get("role")=="user"]
        if correct_ast in user_contents:
            unlock_achievement("asthma_relief")
        abx_opt = "静脉滴注抗生素"
        if abx_opt not in ss.penalty_log:
            unlock_achievement("asthma_no_abx")

    elif case_id == "epiglottitis":
        harm_epi = "强行使用压舌板检查咽喉部"
        if harm_epi not in ss.penalty_log:
            unlock_achievement("epiglottis_protect")
        correct_epi = "保持坐位，吸氧，静脉使用糖皮质激素，准备气管切开/插管"
        user_contents = [m.get("content", "") for m in ss.messages if m.get("role")=="user"]
        if correct_epi in user_contents:
            unlock_achievement("epiglottis_prepare_airway")

# ===================== 成就渲染函数 =====================
def render_mini_achievements(container=st.sidebar):
    """侧边栏迷你展示本局已解锁成就（小缩略图标）"""
    container.markdown("### 🏅本局成就")
    all_ach = get_all_achievement_dict()
    unlocked_list = st.session_state.get("achievements", [])
    show_items = [all_ach[aid] for aid in unlocked_list if aid in all_ach]
    if len(show_items) == 0:
        container.caption("暂未解锁成就")
        return
    html_parts = []
    # 只展示最近6个
    for item in show_items[-6:]:
        html_parts.append(f'<span title="{item["desc"]}" style="font-size:22px;margin:0 2px;">{item["icon"]}</span>')
    container.markdown("".join(html_parts), unsafe_allow_html=True)


def render_achievement_gallery_page():
    """独立页面：完整成就展览馆"""
    st.markdown('<div style="text-align:center;font-size:40px;font-weight:bold;color:#00b4d8;margin:20px 0;">🏅成就展览馆</div>',unsafe_allow_html=True)
    unlocked_set = set(st.session_state.get("achievements", []))
    all_dict = get_all_achievement_dict()

    st.subheader("📌通用成就（全部病例）")
    cols_com = st.columns(3)
    for idx,(aid,ach) in enumerate(COMMON_ACHIEVEMENTS.items()):
        with cols_com[idx%3]:
            is_unlock = aid in unlocked_set
            icon_style = "" if is_unlock else "filter:grayscale(1) opacity:0.3;"
            bg = "rgba(0,180,216,0.15)" if is_unlock else "rgba(80,80,80,0.08)"
            border = "#00b4d8" if is_unlock else "#444444"
            st.markdown(f"""
            <div title="{ach['desc']}" style="text-align:center;background:{bg};border:1px solid {border};border-radius:12px;padding:12px 8px;margin-bottom:10px;">
                <div style="font-size:30px;{icon_style}">{ach['icon']}</div>
                <div style="font-weight:bold;margin-top:6px;">{ach['name']}</div>
                <div style="font-size:12px;opacity:0.7;margin-top:4px;">{ach['desc']}</div>
            </div>""",unsafe_allow_html=True)

    st.divider()
    st.subheader("🩺病例专属成就")
    case_name_map = {
        "laryngitis":"2岁患儿的犬吠声｜急性感染性喉炎",
        "foreign_body":"1岁患儿的剧烈呛咳｜气道异物",
        "asthma":"4岁患儿的反复喘息｜支气管哮喘",
        "epiglottitis":"3岁患儿的高热与流涎｜急性会厌炎"
    }
    for cid,ach_dict in CASE_SPECIAL_ACHIEVEMENTS.items():
        st.markdown(f"#### 📂 {case_name_map[cid]}")
        cols_case = st.columns(2)
        for i,(aid,ach) in enumerate(ach_dict.items()):
            with cols_case[i%2]:
                is_unlock = aid in unlocked_set
                icon_style = "" if is_unlock else "filter:grayscale(1) opacity:0.3;"
                bg = "rgba(230,57,70,0.12)" if is_unlock else "rgba(80,80,80,0.08)"
                border = "#e63946" if is_unlock else "#444444"
                st.markdown(f"""
                <div title="{ach['desc']}" style="text-align:center;background:{bg};border:1px solid {border};border-radius:12px;padding:12px 8px;margin-bottom:10px;">
                    <div style="font-size:30px;{icon_style}">{ach['icon']}</div>
                    <div style="font-weight:bold;margin-top:6px;">{ach['name']}</div>
                    <div style="font-size:12px;opacity:0.7;margin-top:4px;">{ach['desc']}</div>
                </div>""",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

    st.divider()
    if st.button("⬅️ 返回主菜单"):
        st.session_state.screen = "start"
        st.rerun()
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
    # 根据病情度计算生命体征
    if disease_progress < 40:
        hr, spo2, rr, status = 120, 97, 28, "normal"
    elif disease_progress < 70:
        hr, spo2, rr, status = 140, 93, 35, "warn"
    elif disease_progress < 90:
        hr, spo2, rr, status = 160, 88, 45, "danger"
    else:
        hr, spo2, rr, status = 180, 82, 55, "critical"

    # 报警颜色
    if status == "normal":
        hr_color = "#22ff88"
    elif status == "warn":
        hr_color = "#ffcc00"
    else:
        hr_color = "#ff3b3b"

    html_code = f"""
<div style="width:100%; min-height:340px; background:#000808; border:2px solid #00cc66; border-radius:14px; overflow:hidden; position:relative; box-shadow:0 0 18px rgba(0,255,120,0.25);">
    <canvas id="ecgCanvas" width="800" height="340" style="position:absolute; left:0; top:0; width:100%; height:100%;"></canvas>
    <div style="position:absolute; top:12px; right:14px; font-family:'Courier New',monospace; background:rgba(0,0,0,0.65); padding:10px 14px; border-radius:10px; z-index:3; border:1px solid rgba(0,255,120,0.3);">
        <div style="font-size:11px;color:#88aaff;letter-spacing:2px;">HR</div>
        <div style="font-size:34px;font-weight:bold;color:{hr_color};text-shadow:0 0 10px {hr_color};">{hr}</div>
        <div style="font-size:11px;color:#88aaff;letter-spacing:2px;margin-top:8px;">SpO₂</div>
        <div style="font-size:34px;font-weight:bold;color:#22ff88;text-shadow:0 0 10px #22ff88;">{spo2}</div>
        <div style="font-size:11px;color:#88aaff;letter-spacing:2px;margin-top:8px;">RR</div>
        <div style="font-size:34px;font-weight:bold;color:#22ff88;text-shadow:0 0 10px #22ff88;">{rr}</div>
    </div>
    <div style="position:absolute; left:8px; top:8px; color:#39ff66; font-family:Courier New; font-size:13px; z-index:2;">
        ECGⅡ<br>PLETH<br>RESP
    </div>
</div>
<script>
if(window._ecgTimer) clearInterval(window._ecgTimer);
const canvas = document.getElementById('ecgCanvas');
if(!canvas) return;
const ctx = canvas.getContext('2d');
const W = canvas.width;
const H = canvas.height;
const lineGap = H / 3;
let offset = 0;
const speed = 0.04;
const HR = {hr};
const RR = {rr};

// ECG QRS波形
function getEcgVal(t) {{
    const cycle = 60 / HR;
    const p = (t % cycle) / cycle;
    let v = 0;
    if(p>0.02 && p<0.12) v += 0.18 * Math.exp(-Math.pow((p-0.07)/0.025,2));
    if(p>0.14 && p<0.18) v -= 0.22 * Math.exp(-Math.pow((p-0.16)/0.008,2));
    if(p>0.18 && p<0.23) v += 1.25 * Math.exp(-Math.pow((p-0.205)/0.007,2));
    if(p>0.23 && p<0.28) v -= 0.30 * Math.exp(-Math.pow((p-0.25)/0.012,2));
    if(p>0.32 && p<0.55) v += 0.32 * Math.exp(-Math.pow((p-0.42)/0.045,2));
    return v;
}}
// 血氧容积波
function getPlethVal(t) {{
    return 0.4 * Math.sin(2 * Math.PI * (HR/60) * t);
}}
// 呼吸波
function getRespVal(t) {{
    return 0.4 * Math.sin(2 * Math.PI * (RR/60) * t);
}}

function drawGrid() {{
    ctx.fillStyle="#000808";
    ctx.fillRect(0,0,W,H);
    ctx.strokeStyle="rgba(0,120,60,0.18)";
    ctx.lineWidth=0.5;
    for(let x=0;x<W;x+=24){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}}
    for(let y=0;y<H;y+=24){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}
    ctx.strokeStyle="rgba(0,200,100,0.35)";
    ctx.lineWidth=0.8;
    for(let x=0;x<W;x+=120){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(W,y);ctx.stroke();}}
    for(let y=0;y<H;y+=120){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}}
}}

function drawWave() {{
    // 通道1 ECG
    ctx.beginPath();
    ctx.strokeStyle="#22ff88";
    ctx.lineWidth=2.2;
    ctx.shadowColor="#22ff88";
    ctx.shadowBlur=8;
    const y0 = lineGap * 0.5;
    for(let x=0;x<W;x++){{
        const t = (x * 0.0025) + offset;
        const val = getEcgVal(t);
        const y = y0 - val * lineGap * 0.35;
        if(x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }}
    ctx.stroke();

    // 通道2 PLETH
    ctx.beginPath();
    ctx.strokeStyle="#22ff88";
    ctx.lineWidth=1.8;
    ctx.shadowBlur=6;
    const y1 = lineGap * 1.5;
    for(let x=0;x<W;x++){{
        const t = (x * 0.0025) + offset;
        const val = getPlethVal(t);
        const y = y1 - val * lineGap * 0.35;
        if(x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }}
    ctx.stroke();

    // 通道3 RESP
    ctx.beginPath();
    ctx.strokeStyle="#22ff88";
    ctx.lineWidth=1.8;
    ctx.shadowBlur=6;
    const y2 = lineGap * 2.5;
    for(let x=0;x<W;x++){{
        const t = (x * 0.0025) + offset;
        const val = getRespVal(t);
        const y = y2 - val * lineGap * 0.35;
        if(x===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }}
    ctx.stroke();
    ctx.shadowBlur=0;
}}

function render() {{
    drawGrid();
    drawWave();
    offset += speed;
}}
render();
window._ecgTimer = setInterval(render, 35);
</script>
"""
    components.html(html_code, height=360, scrolling=False)
# ================= 新增：病情进度夹紧工具函数 =================
def adjust_disease_progress(delta:int):
    ss = st.session_state
    ss.disease_progress = max(0, min(100, ss.disease_progress + delta))
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
        .status-badge { background: rgba(255, 255, 255, 0.05); border-radius: 14px; padding: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }
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

# ===================== 【成就展览馆路由，放在最前面！】 =====================
if st.session_state.screen == "achievement_gallery":
    render_achievement_gallery_page()
    st.stop()

st.sidebar.markdown("### 🎨 主题设置")
if st.sidebar.button("☀️ 白天模式" if st.session_state.theme == "night" else "🌙 夜晚模式"):
    st.session_state.theme = "day" if st.session_state.theme == "night" else "night"; st.rerun()

# 侧边栏新增成就展览馆按钮
st.sidebar.markdown("---")
if st.sidebar.button("🏅 成就展览馆"):
    st.session_state.screen = "achievement_gallery"
    st.rerun()

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
            st.session_state.player_name = st.session_state.player_name
            st.session_state.screen = "select_case"
            st.rerun()
