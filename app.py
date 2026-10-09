import streamlit as st
import os
import re
import random
import time
import json
import pandas as pd
from openai import OpenAI
import plotly.graph_objects as go
from collections import Counter

# ================= 1. 配置区 =================
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

ALL_CLUES = ["犬吠样咳嗽", "夜间加重", "吸气性喉鸣", "白天感冒史", "三凹征"]

try:
    API_KEY = st.secrets["ZHIPU_API_KEY"]
except Exception:
    API_KEY = "sk-你的真实智谱AI密钥"  # 本地测试时替换

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 2. 页面初始化与深色医疗风UI =================
st.set_page_config(page_title="急诊室疑云：2岁患儿的犬吠声", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f1c2e 0%, #1a2a42 50%, #0f1c2e 100%); color: #e2e8f0; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    .game-main-title { text-align: center; font-size: 44px; font-weight: 900; background: linear-gradient(90deg, #63b3ed, #f56565, #63b3ed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-top: 20px; margin-bottom: 10px; letter-spacing: 3px; }
    .game-sub-title { text-align: center; font-size: 18px; color: #a0aec0; margin-bottom: 40px; }
    
    .mode-card { padding: 30px; border-radius: 20px; margin-bottom: 20px; backdrop-filter: blur(10px); transition: all 0.3s ease; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
    .mode-card:hover { transform: translateY(-8px); box-shadow: 0 16px 48px rgba(0,0,0,0.6); }
    .easy-card { background: linear-gradient(145deg, rgba(72, 187, 120, 0.15), rgba(72, 187, 120, 0.05)); border: 1px solid rgba(72, 187, 120, 0.4); }
    .hard-card { background: linear-gradient(145deg, rgba(229, 62, 62, 0.15), rgba(229, 62, 62, 0.05)); border: 1px solid rgba(229, 62, 62, 0.4); }
    .card-title { font-size: 28px; font-weight: bold; text-align: center; margin-bottom: 15px; }
    .easy-title { color: #68d391; } .hard-title { color: #fc8181; }
    .card-desc { font-size: 15px; color: #cbd5e0; line-height: 1.6; margin-bottom: 15px; text-align: center; }
    .feature-list { list-style-type: none; padding-left: 0; }
    .feature-list li { font-size: 14px; color: #e2e8f0; margin-bottom: 10px; padding-left: 24px; position: relative; }
    .feature-list li:before { content: "✔"; position: absolute; left: 0; color: #68d391; font-weight: bold; font-size: 16px; }
    .hard-card .feature-list li:before { content: "⚡"; color: #fc8181; }
    
    .status-bar { display: flex; justify-content: space-around; gap: 15px; margin-bottom: 25px; flex-wrap: wrap; }
    .status-badge { flex: 1; min-width: 150px; background: rgba(255, 255, 255, 0.05); border-radius: 14px; padding: 15px 20px; backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; }
    .status-label { font-size: 12px; color: #a0aec0; margin-bottom: 6px; }
    .status-value { font-size: 22px; font-weight: 800; }
    .status-trust .status-value { color: #63b3ed; } .status-action .status-value { color: #f6e05e; } .status-disease .status-value { color: #f56565; }
    
    .stButton>button { border-radius: 12px; border: 1px solid rgba(99, 179, 237, 0.4); background: linear-gradient(135deg, rgba(99, 179, 237, 0.15), rgba(99, 179, 237, 0.05)); color: #e2e8f0; font-weight: 500; transition: all 0.3s ease; padding: 12px 20px; }
    .stButton>button:hover { background: linear-gradient(135deg, rgba(99, 179, 237, 0.3), rgba(99, 179, 237, 0.1)); border: 1px solid rgba(99, 179, 237, 0.7); box-shadow: 0 0 20px rgba(99, 179, 237, 0.3); }
    
    .vital-item { background: rgba(255, 255, 255, 0.03); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; border-left: 3px solid #63b3ed; }
    .vital-label { font-size: 11px; color: #a0aec0; } .vital-value { font-size: 17px; font-weight: 700; color: #e2e8f0; }
    
    .clue-item { background: rgba(72, 187, 120, 0.1); border-left: 3px solid #68d391; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 14px; color: #e2e8f0; }
    .clue-locked { background: rgba(255, 255, 255, 0.03); border-left: 3px solid #4a5568; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 14px; color: #718096; }
    .stAlert { background: rgba(255, 255, 255, 0.05); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ================= 3. 游戏模式选择 =================
if "game_mode" not in st.session_state:
    st.markdown('<div class="game-main-title">🏥 急诊室疑云</div>', unsafe_allow_html=True)
    st.markdown('<div class="game-sub-title">2岁患儿的犬吠声 · 请选择你的游戏难度</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""<div class="mode-card easy-card"><div class="card-title easy-title">🌱 简单模式</div><div class="card-desc">适合新手。拥有充足的行动点，系统提供详细的引导和提示。</div><ul class="feature-list"><li>5 个行动点</li><li>初始信任值 40</li><li>明确的任务目标和操作提示</li><li>第四幕抢救限时 90 秒</li></ul></div>""", unsafe_allow_html=True)
        if st.button("开始简单模式", use_container_width=True, type="primary"):
            st.session_state.game_mode = "easy"; st.rerun()
    with col2:
        st.markdown("""<div class="mode-card hard-card"><div class="card-title hard-title">🔥 困难模式</div><div class="card-desc">挑战极限。模拟真实急诊室的极端压力。</div><ul class="feature-list"><li>3 个行动点</li><li>初始信任值 20</li><li>无任务提示，盲盒线索</li><li>第四幕抢救限时 60 秒</li></ul></div>""", unsafe_allow_html=True)
        if st.button("开始困难模式挑战", use_container_width=True, type="primary"):
            st.session_state.game_mode = "hard"; st.rerun()
    st.stop()

# ================= 4. 状态初始化 =================
mode = st.session_state.game_mode
initial_points = 5 if mode == "easy" else 3
initial_trust = 40 if mode == "easy" else 20

st.sidebar.markdown("### 📱 显示设置")
if "view_mode" not in st.session_state: st.session_state.view_mode = "desktop"
if st.sidebar.button("切换手机/电脑视图"):
    st.session_state.view_mode = "mobile" if st.session_state.view_mode == "desktop" else "desktop"; st.rerun()
st.sidebar.caption(f"当前视图：{'电脑端（三栏）' if st.session_state.view_mode == 'desktop' else '手机端（单栏）'}")

st.sidebar.markdown("### 👨‍🏫 教师入口")
if st.sidebar.button("📊 打开教师仪表盘"): st.session_state.page = "teacher"
if st.sidebar.button("🎮 返回游戏"): st.session_state.page = "game"
if "page" not in st.session_state: st.session_state.page = "game"
RECORDS_FILE = "game_records.json"

if st.session_state.page == "teacher":
    st.title("📊 儿科急诊模拟器 · 教师仪表盘")
    if not os.path.exists(RECORDS_FILE): st.warning("暂无数据。"); st.stop()
    with open(RECORDS_FILE, "r", encoding="utf-8") as f: records = json.load(f)
    if len(records) == 0: st.warning("暂无数据。"); st.stop()
    c1, c2, c3 = st.columns(3)
    c1.metric("总测试人次", len(records))
    c2.metric("班级平均分", f"{sum(r['score'] for r in records) / len(records):.1f} / 100")
    c3.metric("最高分", f"{max(r['score'] for r in records)} / 100")
    st.divider()
    st.subheader("🎬 结局分布"); st.bar_chart(pd.DataFrame(Counter(r["title"] for r in records).items(), columns=["结局称号", "人数"]).set_index("结局称号"))
    st.subheader("❌ 误诊方向分布"); st.bar_chart(pd.DataFrame(Counter(r.get("diagnosis", "未选择") for r in records).items(), columns=["诊断选择", "人数"]).set_index("诊断选择"))
    st.subheader("⚠️ 最常见操作失误 Top 5")
    all_penalties = [p for r in records for p in r.get("penalties", [])]
    if all_penalties: st.bar_chart(pd.DataFrame(Counter(all_penalties).most_common(5), columns=["失误操作", "频次"]).set_index("失误操作"))
    else: st.success("🎉 目前没有任何失误操作记录！")
    st.stop()

defaults = {
    "messages": [{"role": "assistant", "content": "医生！您快看看我家小雨！她半夜突然咳得像小狗叫一样，嗓子也哑了，我怎么哄都不行……白天就是有点流鼻涕，我给她喝了点感冒药，怎么会这样啊！"}],
    "trust_score": initial_trust, "action_points": initial_points, "disease_progress": 30, "game_over": False, "time_period": 1,
    "unlocked_clues": [], "final_evaluation": None, "decision_made": {}, "penalty_log": [], "respiratory_action_this_period": False,
    "crisis_actions": [], "crisis_correct_count": 0, "final_score": 0, "final_title": "", "local_reply_cache": [],
    "score_inquiry": 0, "score_diagnosis": 0, "score_emergency": 0, "score_empathy": 0, "diagnosis_made": None,
    "diagnosis_processed": False, "auscultation_mode": False, "auscultation_completed": False, "physical_exam_done": {},
    "achievements": [], "first_act_clue": False, "excluded_diseases": [], "show_dog_cough_img": False, "show_stridor_img": False,
    "show_depression_img": False, "comm_made": False, "last_action_time": time.time(), "crisis_start_time": time.time(), "record_saved": False
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

def render_vital(label, value): st.markdown(f'<div class="vital-item"><div class="vital-label">{label}</div><div class="vital-value">{value}</div></div>', unsafe_allow_html=True)
def restart_game():
    for key in list(st.session_state.keys()): del st.session_state[key]
    st.rerun()

# ================= 5. 剧情配置区 =================
if mode == "easy":
    task_text = "任务目标：安抚家长情绪，通过问诊了解咳嗽的声音特征和发病时间规律。"
    hint_text = "💡 提示：先共情安抚（如“别急，送来得及时”），再切入问诊。"
else:
    task_text = "急诊室气氛紧张，家长情绪极度不稳定，患儿病情不明，你需要迅速做出判断。"
    hint_text = "⚠️ 困难模式：无系统提示。请依靠临床经验，自主发掘病情。"

SCENARIO_DATA = {
    1: {"time": "凌晨 2:00", "title": "🎬 第一幕：急诊室初遇", "task": task_text, "hint": hint_text, "mode": "free"},
    2: {"time": "凌晨 2:30", "title": "🎬 第二幕：初步判断", "task": "任务目标：是否立刻给患儿开检查？请做出你的临床决策。", "hint": "", "mode": "decision"},
    3: {"time": "凌晨 2:45", "title": "🎬 第三幕：迷雾重重", "task": "任务目标：鉴别诊断。询问呼吸情况，并进行体格检查。", "hint": "", "mode": "free"},
    4: {"time": "凌晨 3:00", "title": "🎬 第四幕：生死时速", "task": "突发事件！患儿出现吸气性呼吸困难加重，面色发绀，SpO₂降至88%！请立即处理！", "hint": "", "mode": "crisis"},
    5: {"time": "凌晨 3:15", "title": "🎬 第五幕：医患沟通", "task": "患儿病情暂时稳定，但需要住院观察。家长情绪崩溃，你该如何沟通？", "hint": "", "mode": "communication"},
    6: {"time": "凌晨 3:30", "title": "🎬 第六幕：带教老师介入", "task": "带教老师到场，询问你刚才的处理思路。你该如何回应？", "hint": "", "mode": "decision"},
    7: {"time": "凌晨 4:00", "title": "🎬 第七幕：结案复盘", "task": "任务目标：提交你的最终诊断。", "hint": "", "mode": "end"}
}

DIAGNOSIS_OPTIONS = {
    "A": {"label": "A. 急性感染性喉炎（伴Ⅱ度喉梗阻）", "is_correct": True, "score": 15, "disease_change": -5, "reply": "✅ 诊断正确！你准确识别了犬吠样咳嗽、吸气性喉鸣和夜间加重三大特征。"},
    "B": {"label": "B. 急性会厌炎", "is_correct": False, "score": 0, "disease_change": 25, "reply": "❌ 误诊！患儿没有高热、流涎、吞咽困难，且存在典型的犬吠样咳嗽，不支持会厌炎。"},
    "C": {"label": "C. 气道异物", "is_correct": False, "score": 0, "disease_change": 20, "reply": "❌ 误诊！患儿无突发剧烈呛咳史，且有前驱感冒症状，不支持气道异物。"},
    "D": {"label": "D. 支气管哮喘", "is_correct": False, "score": 0, "disease_change": 15, "reply": "❌ 误诊！患儿表现为吸气性呼吸困难（喉鸣），而非呼气性呼吸困难（哮鸣）。"}
}
AUSCULTATION_OPTIONS = {
    "A": {"label": "A. 吸气性喉鸣（Stridor）", "is_correct": True, "feedback": "✅ 正确！你听到了典型的吸气性喉鸣，这提示上气道梗阻。请继续收集线索，准备给出诊断。"},
    "B": {"label": "B. 呼气性哮鸣音（Wheezing）", "is_correct": False, "feedback": "❌ 错误！你听到的是吸气性喉鸣，而不是呼气性哮鸣音。"},
    "C": {"label": "C. 湿啰音（Crackles）", "is_correct": False, "feedback": "❌ 错误！湿啰音多见于肺炎或肺水肿，与本例不符。"},
    "D": {"label": "D. 呼吸音正常", "is_correct": False, "feedback": "❌ 错误！患儿有明显的呼吸困难，听诊不可能完全正常。"}
}
CRISIS_ACTIONS = {
    "correct_1": {"label": "保持气道通畅：让患儿保持坐位/半坐位，避免哭闹加重喉水肿", "is_correct": True, "feedback": "（你让患儿保持坐位，呼吸稍有缓解）"},
    "correct_2": {"label": "氧疗：面罩吸氧", "is_correct": True, "feedback": "（面罩吸氧后，SpO₂开始缓慢回升）"},
    "correct_3": {"label": "雾化吸入：布地奈德+肾上腺素雾化（关键治疗）", "is_correct": True, "feedback": "（雾化吸入后，喉部水肿明显减轻，喉鸣音减弱）"},
    "correct_4": {"label": "静脉通路：开放静脉，准备糖皮质激素（地塞米松）", "is_correct": True, "feedback": "（静脉通路开放，为后续用药做好准备）"},
    "wrong_1": {"label": "强行按压患儿做咽喉部检查", "is_correct": False, "feedback": "（强行检查刺激喉部，患儿突发喉痉挛，喉鸣音消失，面色青紫！）"},
    "wrong_2": {"label": "使用镇静剂让患儿安静", "is_correct": False, "feedback": "（镇静剂使用后，患儿呼吸变浅变慢，血氧持续下降！）"},
    "wrong_3": {"label": "等待X线结果再处理", "is_correct": False, "feedback": "（等待影像结果的过程中，患儿病情急剧恶化！）"}
}
DECISIONS = {
    1: {
        "prompt": "患儿目前呼吸困难尚可，但声音嘶哑、夜间加重。你打算：",
        "options": {
            "A": {"label": "A. 立即进行床旁喉镜检查，明确喉部情况", "type": "correct", "disease_change": -5, "trust_change": 5, "reply": "（喉镜检查证实喉部黏膜充血水肿，符合喉炎表现）很好，你抓住了关键证据。"},
            "B": {"label": "B. 先观察，开点感冒药让家长回家", "type": "invalid", "disease_change": 25, "trust_change": -15, "reply": "（家长带着孩子离开，2小时后再次抱着孩子冲进来）医生！她更严重了！"},
            "C": {"label": "C. 全套检查：血常规、CRP、胸部CT、心电图、心肌酶谱", "type": "overuse", "disease_change": 15, "trust_change": -10, "reply": "（折腾了1小时，患儿在检查过程中哭闹加剧）医生，能不能先给孩子治治啊？"},
            "D": {"label": "D. 立即使用镇静剂让患儿安静下来配合检查", "type": "harmful", "disease_change": 35, "trust_change": -20, "reply": "（镇静剂使用后，患儿呼吸变浅变慢，血氧开始下降）医生！她怎么睡着了？叫不醒！"}
        }
    },
    2: {
        "prompt": "带教老师赶到，看了一眼监护仪，严肃地问你：刚才紧急处理时，你为什么要这样做？",
        "options": {
            "A": {"label": "A. 承认刚才有失误，详细复盘自己的判断过程，并说明后续改进方向", "type": "correct", "disease_change": 0, "trust_change": 10, "reply": "（带教老师点头）能反思就好。记住，气道急症不能等，处理顺序比检查更重要。"},
            "B": {"label": "B. 沉默不语，只是低头看着监护仪", "type": "invalid", "disease_change": 5, "trust_change": -5, "reply": "（带教老师皱眉）你连自己刚才做了什么都不敢面对吗？"},
            "C": {"label": "C. 把所有责任推给护士，说是护士操作不当", "type": "overuse", "disease_change": 5, "trust_change": -20, "reply": "（带教老师严肃）推卸责任不是一个合格的医生该有的态度。"},
            "D": {"label": "D. 坚持认为自己处理完全正确，不承认任何问题", "type": "harmful", "disease_change": 10, "trust_change": -15, "reply": "（带教老师沉默片刻）你回去把急性喉炎的处理指南抄10遍。"}
        }
    }
}
COMMUNICATION_OPTIONS = {
    "A": {"label": "A. “别哭哭啼啼的，赶紧去办住院手续，别耽误治疗。”", "type": "cold", "trust_change": -10, "score": 0, "reply": "（家长愣住，强忍着眼泪去办手续，但眼神里充满了不信任。）"},
    "B": {"label": "B. “病情很重，喉梗阻随时可能窒息，你们要做好心理准备。”", "type": "scare", "trust_change": -5, "score": 5, "reply": "（家长吓得浑身发抖，崩溃大哭，情绪极度不稳定。）"},
    "C": {"label": "C. “您别自责，来得非常及时。喉炎起病急，但只要及时控制水肿，绝大多数孩子恢复得很好。请您相信我们。”", "type": "empathy", "trust_change": 15, "score": 15, "reply": "（家长擦了擦眼泪，用力点头，情绪逐渐平复，愿意积极配合治疗。）"}
}
ERROR_KNOWLEDGE = {
    "镇静剂": "❌ 错误操作：急性喉梗阻禁用镇静剂！正确做法：保持气道通畅、吸氧、雾化吸入肾上腺素。",
    "CT": "❌ 过度医疗：急性喉炎是临床诊断！不应等待CT或X线结果再处理，搬动患儿会加重喉水肿。",
    "拉肚子": "❌ 无效问诊：偏离主诉！急性喉炎的鉴别诊断核心在于呼吸系统。",
    "听诊：判断错误": "❌ 听诊错误：吸气性喉鸣提示上气道梗阻，常见于急性喉炎。"
}

# ================= 6. 核心逻辑函数 =================
def get_local_reply(prompt, trust_score, period):
    if "小狗" in prompt or "犬吠" in prompt or "狗叫" in prompt: return "她咳起来'空空'的，像小狗叫一样，我从来没听过，吓死我了！"
    if "什么时候" in prompt or "几点" in prompt or "时间" in prompt or "加重" in prompt: return "前天白天有点流鼻涕，半夜突然就咳醒了，大概凌晨1点多，之后就越来越重。"
    if "吸气" in prompt or "呼吸声" in prompt or "喉鸣" in prompt: return "她吸气的时候有'吱吱'的声音，而且胸口凹进去一块，好吓人！"
    if "白天" in prompt or "之前" in prompt or "感冒" in prompt: return "白天就是有点流鼻涕，低烧，我给她喝了点感冒药。怎么晚上突然就成这样了？"
    if "查体" in prompt or "检查" in prompt: return "（配合）您轻点……她胸口这里吸气的时候明显凹进去了。"
    if trust_score < 40: return "医生，您问这些到底有没有用啊？能不能先给孩子吸点氧？"
    return "医生，她嗓子哑了，哭都哭不出声，我该怎么办啊？"

def check_time_pressure():
    if time.time() - st.session_state.last_action_time > 60:
        st.session_state.disease_progress = min(100, st.session_state.disease_progress + 5)
        st.session_state.last_action_time = time.time()
        st.toast("⏰ 患儿病情在等待中加重了！病情 +5", icon="⏳"); return True
    return False

def advance_period(is_decision_phase=False):
    st.session_state.time_period += 1
    st.session_state.last_action_time = time.time()
    if st.session_state.time_period == 4: st.session_state.action_points = 3; st.session_state.crisis_start_time = time.time()
    else: st.session_state.action_points = 5 if mode == "easy" else 3
    if not is_decision_phase:
        st.session_state.disease_progress += (15 if mode == "hard" else 10)
        core_clues = ["犬吠样咳嗽", "夜间加重", "吸气性喉鸣", "三凹征"]
        if not any(c in st.session_state.unlocked_clues for c in core_clues):
            st.session_state.disease_progress += 5
            st.session_state.messages.append({"role": "assistant", "content": "（由于未触及核心线索，患儿病情在不知不觉中加重了...）"})
    st.session_state.respiratory_action_this_period = False
    st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress))
    if st.session_state.disease_progress >= 90:
        st.session_state.messages.append({"role": "assistant", "content": "🚨 监护仪发出刺耳的警报声！患儿出现严重呼吸衰竭征兆！"}); st.session_state.game_over = True
    elif st.session_state.disease_progress >= 70:
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ 患儿出现明显三凹征，喉鸣音加重！情况紧急！"})

# ================= 7. 界面布局 =================
st.markdown(f'<div class="status-bar"><div class="status-badge status-trust"><div class="status-label">❤️ 家长信任值</div><div class="status-value">{st.session_state.trust_score} / 100</div></div><div class="status-badge status-action"><div class="status-label">⏳ 剩余行动点</div><div class="status-value">{st.session_state.action_points} / {5 if mode == "easy" else 3}</div></div><div class="status-badge status-disease"><div class="status-label">⚠️ 病情进展度</div><div class="status-value">{st.session_state.disease_progress} / 100</div></div></div>', unsafe_allow_html=True)

if st.session_state.view_mode == "desktop":
    col_left, col_center, col_right = st.columns([1, 2.5, 1.2])
    
    with col_left:
        st.subheader("📈 实时生命体征")
        st.caption(f"当前时间：{SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[7])['time']}")
        dp = st.session_state.disease_progress
        if dp < 40:
            render_vital("SpO₂", "97%"); render_vital("呼吸", "28 次/分"); render_vital("心率", "120 次/分"); render_vital("意识", "轻度烦躁")
        elif dp < 70:
            render_vital("SpO₂", "93%"); render_vital("呼吸", "35 次/分"); render_vital("心率", "140 次/分"); render_vital("意识", "明显烦躁")
            if mode == "easy": st.warning("⚠️ 出现三凹征，需立即干预")
        elif dp < 90:
            render_vital("SpO₂", "88%"); render_vital("呼吸", "45 次/分"); render_vital("心率", "160 次/分"); render_vital("意识", "发绀、烦躁")
            if mode == "easy": st.error("🚨 喉梗阻加重，随时可能呼吸衰竭")
            st.toast("🚨 生命体征危急！请立即处理！", icon="🚨")
        else:
            render_vital("SpO₂", "82%"); render_vital("呼吸", "55 次/分"); render_vital("心率", "180 次/分"); render_vital("意识", "意识模糊")
            if mode == "easy": st.error("💀 极度危险！随时可能心跳骤停")
            st.toast("💀 患儿濒死！请立即抢救！", icon="💀")
        
        st.divider()
        if not st.session_state.game_over:
            if st.session_state.time_period in [1, 3]:
                if st.session_state.time_period == 3 and st.session_state.action_points <= 2 and not st.session_state.diagnosis_processed:
                    st.warning("⚠️ 问诊结束。请给出初步诊断：")
                    for opt_key, opt in DIAGNOSIS_OPTIONS.items():
                        if st.button(opt["label"], key=f"diag_{opt_key}", use_container_width=True):
                            st.session_state.diagnosis_processed = True; st.session_state.diagnosis_made = opt_key
                            st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + opt["disease_change"]))
                            if opt["is_correct"]:
                                st.session_state.score_diagnosis += opt["score"]
                                if st.session_state.first_act_clue and "一眼定乾坤" not in st.session_state.achievements: st.session_state.achievements.append("一眼定乾坤")
                            else: st.session_state.penalty_log.append(f"诊断：误诊为{opt['label'][:6]}...")
                            st.session_state.messages.append({"role": "user", "content": f"【初步诊断】{opt['label']}"})
                            st.session_state.messages.append({"role": "assistant", "content": opt["reply"]}); st.rerun()
                elif st.session_state.action_points <= 2 and (st.session_state.time_period == 1 or st.session_state.diagnosis_processed):
                    if st.button("▶️ 进入下一幕", use_container_width=True):
                        advance_period(is_decision_phase=False); st.rerun()
                elif st.session_state.action_points > 2:
                    if mode == "easy": st.caption(f"💡 强制问诊阶段：还需 {st.session_state.action_points - 2} 次问诊。")
                    else: st.caption(f"⏳ 剩余问诊次数：{st.session_state.action_points}")
            
            if st.session_state.time_period in [2, 6] and st.session_state.decision_made.get(st.session_state.time_period):
                if st.button("▶️ 继续剧情", use_container_width=True): advance_period(is_decision_phase=True); st.rerun()
            
            if st.session_state.time_period == 5 and not st.session_state.comm_made:
                st.warning("⚠️ 请选择向家长交代病情的方式：")
                for opt_key, opt in COMMUNICATION_OPTIONS.items():
                    if st.button(opt["label"], key=f"comm_{opt_key}", use_container_width=True):
                        st.session_state.comm_made = True; st.session_state.score_empathy += opt["score"]
                        st.session_state.trust_score = max(0, min(100, st.session_state.trust_score + opt["trust_change"]))
                        st.session_state.messages.append({"role": "user", "content": f"【沟通】{opt['label']}"})
                        st.session_state.messages.append({"role": "assistant", "content": opt["reply"]}); st.rerun()
            if st.session_state.time_period == 5 and st.session_state.comm_made:
                if st.button("▶️ 进入下一幕", use_container_width=True): advance_period(is_decision_phase=True); st.rerun()
            
            if st.session_state.time_period == 3 and st.session_state.diagnosis_processed and st.session_state.action_points <= 2:
                if st.button("🚨 突发事件！进入第四幕", use_container_width=True):
                    st.session_state.time_period = 4; st.session_state.action_points = 3 
                    st.session_state.disease_progress = max(70, st.session_state.disease_progress + 10)
                    st.session_state.crisis_start_time = time.time()
                    st.session_state.messages.append({"role": "assistant", "content": "（患儿突然剧烈哭闹，呼吸困难急剧加重）医生！她喘不上气了！"}); st.rerun()
            
            if st.session_state.time_period == 4 and st.session_state.crisis_correct_count >= 2:
                if st.button("✅ 进入第五幕", use_container_width=True):
                    st.session_state.score_emergency += min(25, st.session_state.crisis_correct_count * 10)
                    st.session_state.time_period = 5; st.session_state.action_points = 5 if mode == "easy" else 3
                    st.session_state.messages.append({"role": "assistant", "content": "（经过处理，患儿呼吸逐渐平稳。家长情绪激动...）"}); st.rerun()
        
        st.divider()
        if not st.session_state.game_over and st.session_state.time_period >= 6:
            if st.button("📝 提交诊断，结束游戏", use_container_width=True): st.session_state.game_over = True; st.rerun()
        elif not st.session_state.game_over: st.caption("💡 后期才会开放提交诊断")
        if st.button("🔄 重新开始游戏", use_container_width=True): restart_game()

    with col_right:
        st.subheader("🔍 线索夹")
        with st.expander("🖼️ 体征图库与线索", expanded=True):
            if mode == "hard":
                st.caption("⚠️ 困难模式：线索处于隐藏状态。")
                if len(st.session_state.unlocked_clues) == 0: st.info("暂无已解锁线索。")
                else:
                    for i, clue in enumerate(ALL_CLUES):
                        if clue in st.session_state.unlocked_clues: st.success(f"✅ 线索 {i+1}：{clue}")
                        else: st.text(f"🔒 未知线索 {i+1}/5")
            else:
                if len(st.session_state.unlocked_clues) == 0: st.info("暂无线索，快去问诊吧！")
                else:
                    for clue in ALL_CLUES:
                        if clue in st.session_state.unlocked_clues: st.success(f"✅ {clue}")
                        else: st.text(f"🔒 未知线索")
            if "犬吠样咳嗽" in st.session_state.unlocked_clues and os.path.exists("dog_cough.jpg"): st.image("dog_cough.jpg", caption="犬吠样咳嗽特征", width=300)
            if "吸气性喉鸣" in st.session_state.unlocked_clues and os.path.exists("stridor.jpg"): st.image("stridor.jpg", caption="吸气性喉鸣听诊波形", width=300)
            if "三凹征" in st.session_state.unlocked_clues and os.path.exists("three_depressions.jpg"): st.image("three_depressions.jpg", caption="三凹征", width=350)
        if st.session_state.penalty_log:
            st.divider(); st.caption("📝 操作记录")
            for log in st.session_state.penalty_log: st.warning(f"⚠️ {log}")
    
    with col_center:
        if not st.session_state.game_over:
            current_scenario = SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[7])
            st.title(current_scenario["title"]); st.info(f"**{current_scenario['task']}**\n\n{current_scenario['hint']}")
        else: st.title("🏥 午夜急诊 · 结案")
        if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] in ["free", "decision", "crisis", "communication"]:
            if check_time_pressure(): pass
        if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
            with st.expander("🩺 体格检查工具箱（点击展开，消耗行动点）", expanded=False):
                if mode == "easy": st.caption("💡 儿科急诊查体原则：先安抚，后检查；先救命，后诊病。")
                tab1, tab2, tab3, tab4 = st.tabs(["👁️ 视诊", "👂 听诊", "🖐️ 触诊", "🥁 叩诊"])
                with tab1:
                    if st.button("观察呼吸系统", key="vis_resp"):
                        if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("vis_resp"):
                            st.session_state.action_points -= 1; st.session_state.physical_exam_done["vis_resp"] = True
                            st.session_state.messages.append({"role": "assistant", "content": "（你观察到：吸气时胸骨上窝、锁骨上窝明显凹陷，三凹征阳性！）"})
                            if "三凹征" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("三凹征")
                            st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5); st.rerun()
                with tab2:
                    if st.button("喉部听诊（音频判断）", key="aus_larynx"):
                        if st.session_state.action_points > 0 and not st.session_state.auscultation_completed:
                            st.session_state.action_points -= 1; st.session_state.auscultation_mode = True; st.rerun()
                with tab3:
                    if st.button("腹部触诊", key="pal_abdomen"):
                        if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("pal_abdomen"):
                            st.session_state.action_points -= 1; st.session_state.physical_exam_done["pal_abdomen"] = True
                            st.session_state.messages.append({"role": "assistant", "content": "（患儿因缺氧哭闹剧烈，无法配合触诊。）"})
                            st.session_state.disease_progress = min(100, st.session_state.disease_progress + 5); st.toast("⚠️ 查体不配合，病情 +5", icon="📈"); st.rerun()
                with tab4:
                    if st.button("肺部叩诊", key="per_lung"):
                        if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("per_lung"):
                            st.session_state.action_points -= 1; st.session_state.physical_exam_done["per_lung"] = True
                            st.session_state.messages.append({"role": "assistant", "content": "（患儿哭闹，无法配合叩诊。）"})
                            st.session_state.disease_progress = min(100, st.session_state.disease_progress + 5); st.rerun()
        if st.session_state.auscultation_mode:
            st.warning("🩺 你戴上听诊器，请仔细听诊患儿的呼吸音：")
            if os.path.exists("stridor.mp3"): st.audio("stridor.mp3", format="audio/mp3")
            else: st.info("🔇 未找到本地音频文件 'stridor.mp3'。请想象：吸气时出现高调、粗糙的'吱吱'声...")
            if os.path.exists("stridor.jpg"): st.image("stridor.jpg", caption="吸气性喉鸣音波形图", width=350)
            st.write("请判断你听到的是什么呼吸音：")
            for opt_key, opt in AUSCULTATION_OPTIONS.items():
                if st.button(opt["label"], key=f"aus_{opt_key}", use_container_width=True):
                    st.session_state.auscultation_mode = False; st.session_state.auscultation_completed = True
                    st.session_state.messages.append({"role": "user", "content": f"【听诊】{opt['label']}"})
                    st.session_state.messages.append({"role": "assistant", "content": opt["feedback"]})
                    if opt["is_correct"]:
                        st.session_state.score_diagnosis = min(30, st.session_state.score_diagnosis + 10)
                        st.session_state.disease_progress = max(0, st.session_state.disease_progress - 10)
                        if "吸气性喉鸣" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("吸气性喉鸣")
                        st.toast("✅ 听诊正确！解锁线索【吸气性喉鸣】，病情缓解 -10", icon="📉")
                    else:
                        st.session_state.penalty_log.append("听诊：判断错误")
                        st.session_state.disease_progress = min(100, st.session_state.disease_progress + 10)
                        st.toast("❌ 听诊错误！病情加重 +10", icon="🚨")
                    st.rerun()
        else:
            with st.container(height=450):
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]): st.write(msg["content"])
            if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "decision":
                decision_key = 1 if st.session_state.time_period == 2 else 2
                if not st.session_state.decision_made.get(st.session_state.time_period):
                    decision = DECISIONS[decision_key]
                    if mode == "easy": st.warning(f"⚠️ {decision['prompt']}")
                    else: st.warning("⚠️ 请立刻做出你的临床决策：")
                    for opt_key, opt in decision["options"].items():
                        if st.button(opt["label"], key=f"dec_{st.session_state.time_period}_{opt_key}", use_container_width=True):
                            st.session_state.decision_made[st.session_state.time_period] = opt_key
                            st.session_state.messages.append({"role": "user", "content": f"【决策】{opt['label']}"})
                            st.session_state.messages.append({"role": "assistant", "content": opt["reply"]})
                            st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + opt["disease_change"]))
                            st.session_state.trust_score = max(0, min(100, st.session_state.trust_score + opt["trust_change"]))
                            if opt["type"] != "correct": st.session_state.penalty_log.append(f"决策失误：{opt['label'][:8]}...")
                            st.rerun()
            elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "crisis":
                time_limit = 60 if mode == "hard" else 90
                elapsed = time.time() - st.session_state.crisis_start_time
                time_left = max(0, time_limit - int(elapsed))
                st.warning("⚠️ 请从下列操作中选择紧急处理方案（每个操作消耗1个行动点，共3个行动点）：")
                st.progress(max(0.0, 1.0 - (elapsed / time_limit)))
                if time_left <= 15:
                    st.error(f"⏰ 紧急抢救倒计时：{time_left} 秒！时间不多了，请立即决断！")
                    if os.path.exists("alarm.mp3"): st.audio("alarm.mp3", format="audio/mp3", autoplay=True)
                else: st.info(f"⏰ 紧急抢救倒计时：{time_left} 秒")
                if time_left <= 0:
                    st.session_state.disease_progress = 100; st.session_state.game_over = True
                    st.session_state.messages.append({"role": "assistant", "content": "（抢救超时！患儿出现严重窒息，心跳骤停！）"}); st.rerun()
                if st.session_state.action_points > 0:
                    for act_key, act in CRISIS_ACTIONS.items():
                        if act_key not in st.session_state.crisis_actions:
                            if st.button(act["label"], key=f"crisis_{act_key}", use_container_width=True):
                                st.session_state.action_points -= 1; st.session_state.crisis_actions.append(act_key)
                                st.session_state.messages.append({"role": "user", "content": act["label"]})
                                if act["is_correct"]:
                                    st.session_state.crisis_correct_count += 1
                                    st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress - 15))
                                    st.session_state.messages.append({"role": "assistant", "content": act["feedback"] + "（患儿面色稍有缓解）"})
                                    st.toast("✅ 病情缓解 -15", icon="📉")
                                else:
                                    st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + 25))
                                    st.session_state.messages.append({"role": "assistant", "content": act["feedback"] + "（患儿面色更加青紫！）"})
                                    st.session_state.penalty_log.append(f"紧急处理失误：{act['label'][:8]}...")
                                    st.toast("❌ 病情加重 +25", icon="🚨")
                                st.rerun()
                if st.session_state.action_points <= 0 or len(st.session_state.crisis_actions) >= 3:
                    st.divider()
                    if st.session_state.crisis_correct_count >= 2: st.success(f"✅ 你做对了 {st.session_state.crisis_correct_count} 项正确操作！请点击左侧『进入第五幕』。")
                    else:
                        st.error(f"❌ 正确操作不足2项。患儿病情急剧恶化，触发 Bad Ending！")
                        st.session_state.disease_progress = min(100, st.session_state.disease_progress + 20); st.session_state.game_over = True; st.rerun()
            elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
                if st.session_state.time_period == 3 and st.session_state.action_points <= 2 and not st.session_state.diagnosis_processed:
                    if mode == "easy": st.info("👉 请根据现有线索，做出初步诊断。")
                    else: st.info("👉 时间紧迫，请根据你的专业判断给出诊断。")
                else:
                    if prompt := st.chat_input("请输入你的问诊、查体或检查操作..."):
                        if st.session_state.action_points <= 0: st.warning("行动点已用完！请点击左侧『进入下一幕』。")
                        else:
                            st.session_state.action_points -= 1; st.session_state.last_action_time = time.time()
                            st.session_state.messages.append({"role": "user", "content": prompt})
                            if any(k in prompt for k in ["听诊", "听呼吸", "听肺"]):
                                if not st.session_state.auscultation_completed: st.session_state.auscultation_mode = True; st.rerun()
                                else: st.session_state.messages.append({"role": "assistant", "content": "（你已经听过了，患儿现在很烦躁，不宜反复听诊。）"}); st.rerun()
                            if st.session_state.time_period == 1 and any(k in prompt for k in ["小狗", "犬吠", "狗叫", "咳嗽声音"]): st.session_state.first_act_clue = True
                            if any(k in prompt for k in ["流口水", "吞咽", "会厌"]):
                                if "会厌炎" not in st.session_state.excluded_diseases: st.session_state.excluded_diseases.append("会厌炎")
                            if any(k in prompt for k in ["呛", "异物", "吃东西"]):
                                if "异物" not in st.session_state.excluded_diseases: st.session_state.excluded_diseases.append("异物")
                            if any(k in prompt for k in ["喘", "哮喘", "过敏"]):
                                if "哮喘" not in st.session_state.excluded_diseases: st.session_state.excluded_diseases.append("哮喘")
                            if len(st.session_state.excluded_diseases) == 3:
                                if "千金难买早知道" not in st.session_state.achievements:
                                    st.session_state.achievements.append("千金难买早知道"); st.toast("🎉 解锁成就：千金难买早知道！", icon="🏅")
                            trust_change = 0; disease_change = 5; clue = "无"
                            if any(kw in prompt for kw in ["别急", "送来得及时", "别怕", "我帮您", "冷静", "理解", "放心"]): trust_change = 10; disease_change = 0; st.session_state.score_empathy = min(15, st.session_state.score_empathy + 5)
                            elif any(kw in prompt for kw in ["怎么才", "你怎么", "搞什么", "麻烦", "快点"]): trust_change = -15; disease_change = 10
                            if any(k in prompt for k in ["小狗", "犬吠", "狗叫", "咳嗽声音", "什么样的咳"]): clue = "犬吠样咳嗽"; disease_change = -5; st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8); st.session_state.show_dog_cough_img = True
                            elif any(k in prompt for k in ["什么时候", "几点", "时间", "加重", "晚上", "半夜", "凌晨"]): clue = "夜间加重"; disease_change = -5; st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8)
                            elif any(k in prompt for k in ["吸气", "呼吸声", "喉鸣", "喘气声"]): clue = "吸气性喉鸣"; disease_change = -10; st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8); st.session_state.show_stridor_img = True
                            elif any(k in prompt for k in ["白天", "之前", "前几天", "感冒"]): clue = "白天感冒史"; disease_change = 0; st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 6)
                            elif any(k in prompt for k in ["拉肚子", "皮疹", "呕吐"]): disease_change = 15; st.toast("⚠️ 无效问诊！病情加重！", icon="⚠️")
                            st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + disease_change))
                            history_text = "".join([f"{'医生' if m['role']=='user' else '家属'}: {m['content']}\n" for m in st.session_state.messages[-6:]])
                            reply_text = None
                            try:
                                tone_prompt = "非常焦虑和自责" if st.session_state.trust_score < 40 else ("有些紧张但配合" if st.session_state.trust_score < 70 else "信任医生并感激")
                                ai_prompt = f"你是2岁急性喉炎患儿的妈妈。当前情绪：{tone_prompt}。事实：前天白天流鼻涕，凌晨1点半突发犬吠样咳嗽。\n【对话历史】\n{history_text}\n【铁律】1.对医生说话。2.句子完整。3.回复50字以内。\n医生刚刚说：'{prompt}'\n请直接回答医生："
                                ai_response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": ai_prompt}], temperature=0.4, timeout=3)
                                reply_text = ai_response.choices[0].message.content.strip()
                            except Exception: reply_text = get_local_reply(prompt, st.session_state.trust_score, st.session_state.time_period)
                            if not reply_text: reply_text = get_local_reply(prompt, st.session_state.trust_score, st.session_state.time_period)
                            bad_phrases = ["宝宝", "妈妈对不起", "咱们要坚强", "乖", "妈妈在", "吓坏妈妈", "那么厉害", "谢谢你啊"]
                            if any(k in reply_text for k in bad_phrases) or len(reply_text) < 5: reply_text = "医生，呜呜，她咳得喘不上气，您快救救她吧！"
                            if len(reply_text) > 60: reply_text = reply_text[:60] + "..."
                            if any(k in reply_text for k in ["小狗", "犬吠", "狗叫"]):
                                if "犬吠样咳嗽" not in st.session_state.unlocked_clues: clue = "犬吠样咳嗽"
                            if any(k in reply_text for k in ["半夜", "凌晨", "睡着", "夜里"]):
                                if "夜间加重" not in st.session_state.unlocked_clues: clue = "夜间加重"
                            st.session_state.messages.append({"role": "assistant", "content": reply_text})
                            st.session_state.trust_score = max(0, min(100, st.session_state.trust_score + trust_change))
                            if clue != "无" and clue in ALL_CLUES:
                                if clue not in st.session_state.unlocked_clues:
                                    st.session_state.unlocked_clues.append(clue); st.toast(f"🎉 解锁新线索：{clue}", icon="🔍")
                            if st.session_state.disease_progress >= 100:
                                st.session_state.game_over = True
                                if "反面教材" not in st.session_state.achievements: st.session_state.achievements.append("反面教材")
                            st.rerun()

# 手机端单栏视图
else:
    if not st.session_state.game_over:
        current_scenario = SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[7])
        st.title(current_scenario["title"]); st.info(f"**{current_scenario['task']}**\n\n{current_scenario['hint']}")
    else: st.title("🏥 午夜急诊 · 结案")
    if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
        with st.expander("🩺 体格检查工具箱（点击展开，消耗行动点）", expanded=False):
            tab1, tab2 = st.tabs(["👁️ 视诊", "👂 听诊"])
            with tab1:
                if st.button("观察呼吸系统", key="m_vis_resp"):
                    if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("vis_resp"):
                        st.session_state.action_points -= 1; st.session_state.physical_exam_done["vis_resp"] = True
                        st.session_state.messages.append({"role": "assistant", "content": "（你观察到：吸气时胸骨上窝、锁骨上窝明显凹陷，三凹征阳性！）"})
                        if "三凹征" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("三凹征")
                        st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5); st.rerun()
            with tab2:
                if st.button("喉部听诊（音频判断）", key="m_aus_larynx"):
                    if st.session_state.action_points > 0 and not st.session_state.auscultation_completed:
                        st.session_state.action_points -= 1; st.session_state.auscultation_mode = True; st.rerun()
    if st.session_state.auscultation_mode:
        st.warning("🩺 你戴上听诊器，请仔细听诊患儿的呼吸音：")
        if os.path.exists("stridor.mp3"): st.audio("stridor.mp3", format="audio/mp3")
        st.write("请判断你听到的是什么呼吸音：")
        for opt_key, opt in AUSCULTATION_OPTIONS.items():
            if st.button(opt["label"], key=f"m_aus_{opt_key}", use_container_width=True):
                st.session_state.auscultation_mode = False; st.session_state.auscultation_completed = True
                st.session_state.messages.append({"role": "user", "content": f"【听诊】{opt['label']}"})
                st.session_state.messages.append({"role": "assistant", "content": opt["feedback"]})
                if opt["is_correct"]:
                    st.session_state.score_diagnosis = min(30, st.session_state.score_diagnosis + 10)
                    st.session_state.disease_progress = max(0, st.session_state.disease_progress - 10)
                    if "吸气性喉鸣" not in st.session_state.unlocked_clues: st.session_state.unlocked_clues.append("吸气性喉鸣")
                st.rerun()
    else:
        with st.container(height=500):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.write(msg["content"])
        if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
            if prompt := st.chat_input("请输入问诊、查体或检查操作..."):
                if st.session_state.action_points > 0:
                    st.session_state.action_points -= 1; st.session_state.messages.append({"role": "user", "content": prompt}); st.rerun()

# ================= 8. 游戏结算与复盘 =================
if st.session_state.game_over:
    st.divider(); st.header("🩺 带教老师复盘")
    if st.session_state.disease_progress >= 100:
        if "反面教材" not in st.session_state.achievements: st.session_state.achievements.append("反面教材")
    if st.session_state.final_score == 0:
        s_inquiry = st.session_state.score_inquiry; s_diagnosis = st.session_state.score_diagnosis
        s_emergency = st.session_state.score_emergency; s_empathy = min(15, st.session_state.score_empathy + (st.session_state.trust_score * 0.1))
        total = s_inquiry + s_diagnosis + s_emergency + s_empathy; st.session_state.final_score = round(total, 1)
        if st.session_state.final_score >= 90: st.session_state.final_title = "🏆 儿科急诊之光"
        elif st.session_state.final_score >= 70: st.session_state.final_title = "🌟 有潜力的住院医"
        elif st.session_state.final_score >= 50: st.session_state.final_title = "📚 还需回炉重造"
        else: st.session_state.final_title = "😡 小雨妈妈已向医务科投诉"
        if not st.session_state.record_saved:
            record = {"score": st.session_state.final_score, "title": st.session_state.final_title, "diagnosis": st.session_state.diagnosis_made, "penalties": list(set(st.session_state.penalty_log)), "mode": mode}
            data = json.load(open(RECORDS_FILE, "r", encoding="utf-8")) if os.path.exists(RECORDS_FILE) else []
            data.append(record); json.dump(data, open(RECORDS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=4); st.session_state.record_saved = True
    if st.session_state.final_evaluation is None:
        with st.spinner("带教老师正在复盘..."):
            chat_history = "\n".join([f"{'医生' if m['role']=='user' else '家属'}: {m['content']}" for m in st.session_state.messages])
            penalty_text = "\n".join(st.session_state.penalty_log) if st.session_state.penalty_log else "无"
            eval_prompt = f"你是儿科急诊带教老师。请点评这位医学生的表现。\n【状态】信任值:{st.session_state.trust_score} | 病情度:{st.session_state.disease_progress} | 线索:{st.session_state.unlocked_clues}\n【失误】{penalty_text}\n【记录】{chat_history}\n【要求】先肯定优点再指出问题，结合得分给出改进建议，300字左右。"
            try:
                eval_response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": eval_prompt}], temperature=0.7, stream=True, timeout=20)
                st.session_state.final_evaluation = st.write_stream(eval_response)
            except Exception: st.session_state.final_evaluation = "【系统自动评语】你完成了本次急救演练。请在错题本中复习失误点。"
    else: st.info(st.session_state.final_evaluation)
    
    st.divider(); st.subheader("📊 个人能力雷达图")
    s_empathy_final = min(15, st.session_state.score_empathy + (st.session_state.trust_score * 0.1))
    scores = [st.session_state.score_inquiry, st.session_state.score_diagnosis, st.session_state.score_emergency, s_empathy_final]
    fig = go.Figure(); fig.add_trace(go.Scatterpolar(r=[(s / m) * 100 for s, m in zip(scores, [30, 30, 25, 15])], theta=['问诊完整性', '鉴别诊断', '紧急处理', '医患沟通'], fill='toself'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=400, margin=dict(l=40, r=40, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider(); st.subheader("🌳 动态诊断树复盘")
    st.code(f"呼吸困难\n├── 吸气性（喉鸣）\n│   ├── 急性喉炎 {'✅ (已确诊)' if st.session_state.diagnosis_made == 'A' else '❌ (未能确诊)'}\n│   ├── 会厌炎 {'❌ (已排除)' if '会厌炎' in st.session_state.excluded_diseases else '⚠️ (未评估)'}\n│   └── 异物 {'❌ (已排除)' if '异物' in st.session_state.excluded_diseases else '⚠️ (未评估)'}\n└── 呼气性（哮鸣）\n    └── 哮喘 {'❌ (已排除)' if '哮喘' in st.session_state.excluded_diseases else '⚠️ (未评估)'}\n", language="text")
    
    st.divider(); st.subheader("📝 错题本与知识点复盘")
    if not st.session_state.penalty_log: st.success("🎉 完美！你没有任何失误记录！")
    else:
        for log in st.session_state.penalty_log:
            st.warning(f"📌 {log}")
            for key, knowledge in ERROR_KNOWLEDGE.items():
                if key in log: st.info(knowledge); break
    
    st.divider(); st.subheader("🏆 隐藏成就")
    if not st.session_state.achievements: st.write("暂无成就，再接再厉！")
    else:
        for ach in st.session_state.achievements:
            if ach == "一眼定乾坤": st.markdown("🥇 **一眼定乾坤**：第一幕就问出关键症状并成功确诊！")
            elif ach == "千金难买早知道": st.markdown("🥇 **千金难买早知道**：排除了所有高危鉴别诊断！")
            elif ach == "反面教材": st.markdown("🥇 **反面教材**：患儿病情达到极度危险状态，请吸取教训！")
    
    st.divider(); st.subheader("🏅 综合评价")
    st.metric("最终总分", f"{st.session_state.final_score} / 100"); st.markdown(f"### {st.session_state.final_title}")
    if st.session_state.disease_progress >= 100: st.error("结局：Bad Ending。患儿因未及时处理喉梗阻，出现呼吸衰竭。")
    elif st.session_state.trust_score < 40: st.warning("结局：家长因不信任你的沟通，抱着孩子转院了。")
    elif len(st.session_state.penalty_log) > 0: st.warning("结局：Neutral Ending。患儿最终好转，但你的操作存在明显失误。")
    else: st.success("结局：Good Ending！你准确识别了急性喉炎，患儿症状缓解。")
