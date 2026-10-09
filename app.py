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
    /* 🌟 全局深色背景 */
    .stApp {
        background: linear-gradient(135deg, #0f1c2e 0%, #1a2a42 50%, #0f1c2e 100%);
        color: #e2e8f0;
    }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* 🌟 标题 */
    .game-main-title {
        text-align: center; font-size: 44px; font-weight: 900;
        background: linear-gradient(90deg, #63b3ed, #f56565, #63b3ed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-top: 20px; margin-bottom: 10px; letter-spacing: 3px;
        text-shadow: 0 0 30px rgba(99, 179, 237, 0.3);
    }
    .game-sub-title { text-align: center; font-size: 18px; color: #a0aec0; margin-bottom: 40px; letter-spacing: 1px; }
    
    /* 🌟 模式卡片 */
    .mode-card {
        padding: 30px; border-radius: 20px; margin-bottom: 20px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .mode-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 48px rgba(0,0,0,0.6);
    }
    .easy-card {
        background: linear-gradient(145deg, rgba(72, 187, 120, 0.15), rgba(72, 187, 120, 0.05));
        border: 1px solid rgba(72, 187, 120, 0.4);
    }
    .hard-card {
        background: linear-gradient(145deg, rgba(229, 62, 62, 0.15), rgba(229, 62, 62, 0.05));
        border: 1px solid rgba(229, 62, 62, 0.4);
    }
    .card-title { font-size: 28px; font-weight: bold; text-align: center; margin-bottom: 15px; }
    .easy-title { color: #68d391; }
    .hard-title { color: #fc8181; }
    .card-desc { font-size: 15px; color: #cbd5e0; line-height: 1.6; margin-bottom: 15px; text-align: center; }
    .feature-list { list-style-type: none; padding-left: 0; }
    .feature-list li { font-size: 14px; color: #e2e8f0; margin-bottom: 10px; padding-left: 24px; position: relative; }
    .feature-list li:before {
        content: "✔"; position: absolute; left: 0; color: #68d391; font-weight: bold; font-size: 16px;
    }
    .hard-card .feature-list li:before { content: "⚡"; color: #fc8181; }
    
    /* 🌟 顶部状态徽章栏 */
    .status-bar {
        display: flex; justify-content: space-around; gap: 15px;
        margin-bottom: 25px; flex-wrap: wrap;
    }
    .status-badge {
        flex: 1; min-width: 150px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 14px; padding: 15px 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .status-label { font-size: 12px; color: #a0aec0; margin-bottom: 6px; letter-spacing: 1px; }
    .status-value { font-size: 22px; font-weight: 800; color: #e2e8f0; }
    .status-trust .status-value { color: #63b3ed; }
    .status-action .status-value { color: #f6e05e; }
    .status-disease .status-value { color: #f56565; }
    
    /* 🌟 侧边栏样式 */
    .sidebar .sidebar-content { background: rgba(15, 28, 46, 0.95); }
    
    /* 🌟 聊天气泡 */
    div[data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 16px; padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 12px;
        backdrop-filter: blur(5px);
    }
    
    /* 🌟 按钮 */
    .stButton>button {
        border-radius: 12px;
        border: 1px solid rgba(99, 179, 237, 0.4);
        background: linear-gradient(135deg, rgba(99, 179, 237, 0.15), rgba(99, 179, 237, 0.05));
        color: #e2e8f0;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 12px 20px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, rgba(99, 179, 237, 0.3), rgba(99, 179, 237, 0.1));
        border: 1px solid rgba(99, 179, 237, 0.7);
        box-shadow: 0 0 20px rgba(99, 179, 237, 0.3);
    }
    
    /* 🌟 生命体征卡片 */
    .vital-item {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px; padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 3px solid #63b3ed;
    }
    .vital-label { font-size: 11px; color: #a0aec0; }
    .vital-value { font-size: 17px; font-weight: 700; color: #e2e8f0; }
    
    /* 🌟 线索卡 */
    .clue-item {
        background: rgba(72, 187, 120, 0.1);
        border-left: 3px solid #68d391;
        border-radius: 8px; padding: 10px 14px;
        margin-bottom: 8px; font-size: 14px; color: #e2e8f0;
    }
    .clue-locked {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #4a5568;
        border-radius: 8px; padding: 10px 14px;
        margin-bottom: 8px; font-size: 14px; color: #718096;
    }
    
    /* 🌟 信息框 */
    .stAlert { background: rgba(255, 255, 255, 0.05); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ================= 3. 游戏模式选择 =================
if "game_mode" not in st.session_state:
    st.markdown('<div class="game-main-title">🏥 急诊室疑云</div>', unsafe_allow_html=True)
    st.markdown('<div class="game-sub-title">2岁患儿的犬吠声 · 请选择你的游戏难度</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("""
        <div class="mode-card easy-card">
            <div class="card-title easy-title">🌱 简单模式</div>
            <div class="card-desc">适合新手。拥有充足的行动点，系统提供详细的引导和提示。</div>
            <ul class="feature-list">
                <li>5 个行动点</li>
                <li>初始信任值 40</li>
                <li>明确的任务目标和操作提示</li>
                <li>第四幕抢救限时 90 秒</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("开始简单模式", use_container_width=True, type="primary"):
            st.session_state.game_mode = "easy"
            st.rerun()
    with col2:
        st.markdown("""
        <div class="mode-card hard-card">
            <div class="card-title hard-title">🔥 困难模式</div>
            <div class="card-desc">挑战极限。模拟真实急诊室的极端压力。</div>
            <ul class="feature-list">
                <li>3 个行动点</li>
                <li>初始信任值 20</li>
                <li>无任务提示，盲盒线索</li>
                <li>第四幕抢救限时 60 秒</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("开始困难模式挑战", use_container_width=True, type="primary"):
            st.session_state.game_mode = "hard"
            st.rerun()
    st.stop()

# ================= 4. 状态初始化 =================
mode = st.session_state.game_mode
initial_points = 5 if mode == "easy" else 3
initial_trust = 40 if mode == "easy" else 20

# 侧边栏
st.sidebar.markdown("### 📱 显示设置")
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "desktop"
if st.sidebar.button("切换手机/电脑视图"):
    st.session_state.view_mode = "mobile" if st.session_state.view_mode == "desktop" else "desktop"
    st.rerun()
st.sidebar.caption(f"当前视图：{'电脑端（三栏）' if st.session_state.view_mode == 'desktop' else '手机端（单栏）'}")

st.sidebar.markdown("### 👨‍🏫 教师入口")
if st.sidebar.button("📊 打开教师仪表盘"):
    st.session_state.page = "teacher"
if st.sidebar.button("🎮 返回游戏"):
    st.session_state.page = "game"
if "page" not in st.session_state:
    st.session_state.page = "game"

RECORDS_FILE = "game_records.json"

# 教师仪表盘视图
if st.session_state.page == "teacher":
    st.title("📊 儿科急诊模拟器 · 教师仪表盘")
    st.markdown("该面板展示班级同学在游戏中的整体表现。")
    if not os.path.exists(RECORDS_FILE):
        st.warning("暂无数据。请让同学们至少完成一局游戏。")
        st.stop()
    with open(RECORDS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) == 0:
        st.warning("暂无数据。")
        st.stop()
    total_plays = len(records)
    avg_score = sum(r["score"] for r in records) / total_plays
    max_score = max(r["score"] for r in records)
    col1, col2, col3 = st.columns(3)
    col1.metric("总测试人次", total_plays)
    col2.metric("班级平均分", f"{avg_score:.1f} / 100")
    col3.metric("最高分", f"{max_score} / 100")
    st.divider()
    st.subheader("🎬 结局分布")
    title_counts = Counter(r["title"] for r in records)
    title_df = pd.DataFrame(title_counts.items(), columns=["结局称号", "人数"])
    st.bar_chart(title_df.set_index("结局称号"))
    st.subheader("❌ 误诊方向分布")
    diagnosis_map = {"A": "急性喉炎（正确）", "B": "急性会厌炎（误诊）", "C": "气道异物（误诊）", "D": "支气管哮喘（误诊）"}
    diag_counts = Counter(r.get("diagnosis", "未选择") for r in records)
    diag_df = pd.DataFrame([(diagnosis_map.get(k, k), v) for k, v in diag_counts.items()], columns=["诊断选择", "人数"])
    st.bar_chart(diag_df.set_index("诊断选择"))
    st.subheader("⚠️ 最常见操作失误 Top 5")
    all_penalties = []
    for r in records:
        all_penalties.extend(r.get("penalties", []))
    if all_penalties:
        penalty_counts = Counter(all_penalties)
        penalty_df = pd.DataFrame(penalty_counts.most_common(5), columns=["失误操作", "频次"])
        st.bar_chart(penalty_df.set_index("失误操作"))
    else:
        st.success("🎉 目前没有任何失误操作记录！")
    st.stop()

# 游戏状态初始化
defaults = {
    "messages": [{"role": "assistant", "content": "医生！您快看看我家小雨！她半夜突然咳得像小狗叫一样，嗓子也哑了，我怎么哄都不行……白天就是有点流鼻涕，我给她喝了点感冒药，怎么会这样啊！"}],
    "trust_score": initial_trust,
    "action_points": initial_points,
    "disease_progress": 30,
    "game_over": False,
    "time_period": 1,
    "unlocked_clues": [],
    "final_evaluation": None,
    "decision_made": {},
    "penalty_log": [],
    "respiratory_action_this_period": False,
    "crisis_actions": [],
    "crisis_correct_count": 0,
    "final_score": 0,
    "final_title": "",
    "local_reply_cache": [],
    "score_inquiry": 0,
    "score_diagnosis": 0,
    "score_emergency": 0,
    "score_empathy": 0,
    "diagnosis_made": None,
    "diagnosis_processed": False,
    "auscultation_mode": False,
    "auscultation_completed": False,
    "physical_exam_done": {},
    "achievements": [],
    "first_act_clue": False,
    "excluded_diseases": [],
    "show_dog_cough_img": False,
    "show_stridor_img": False,
    "show_depression_img": False,
    "comm_made": False,
    "last_action_time": time.time(),
    "crisis_start_time": time.time(),
    "record_saved": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def render_vital(label, value):
    st.markdown(
        f"""
        <div class="vital-item">
            <div class="vital-label">{label}</div>
            <div class="vital-value">{value}</div>
        </div>
        """, unsafe_allow_html=True
    )

def restart_game():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
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
    "wrong_3": {"label": "等待X线结果再处理", "is_correct": False, "feedback": "（等待影像结果的过程中，患儿病情急剧恶化！）"},
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
    "听诊：判断错误": "❌ 听诊错误：吸气性喉鸣提示上气道梗阻，常见于急性喉炎。",
}

# ================= 6. 核心逻辑函数 =================
def get_local_reply(prompt, trust_score, period):
    if "小狗" in prompt or "犬吠" in prompt or "狗叫" in prompt:
        return "她咳起来'空空'的，像小狗叫一样，我从来没听过，吓死我了！"
    if "什么时候" in prompt or "几点" in prompt or "时间" in prompt or "加重" in prompt:
        return "前天白天有点流鼻涕，半夜突然就咳醒了，大概凌晨1点多，之后就越来越重。"
    if "吸气" in prompt or "呼吸声" in prompt or "喉鸣" in prompt:
        return "她吸气的时候有'吱吱'的声音，而且胸口凹进去一块，好吓人！"
    if "白天" in prompt or "之前" in prompt or "感冒" in prompt:
        return "白天就是有点流鼻涕，低烧，我给她喝了点感冒药。怎么晚上突然就成这样了？"
    if "查体" in prompt or "检查" in prompt:
        return "（配合）您轻点……她胸口这里吸气的时候明显凹进去了。"
    if trust_score < 40:
        return "医生，您问这些到底有没有用啊？能不能先给孩子吸点氧？"
    return "医生，她嗓子哑了，哭都哭不出声，我该怎么办啊？"

def check_time_pressure():
    current_time = time.time()
    elapsed = current_time - st.session_state.last_action_time
    if elapsed > 60:
        st.session_state.disease_progress = min(100, st.session_state.disease_progress + 5)
        st.session_state.last_action_time = current_time
        st.toast("⏰ 患儿病情在等待中加重了！病情 +5", icon="⏳")
        return True
    return False

def advance_period(is_decision_phase=False):
    st.session_state.time_period += 1
    st.session_state.last_action_time = time.time()
    
    if st.session_state.time_period == 4:
        st.session_state.action_points = 3
        st.session_state.crisis_start_time = time.time()
    else:
        st.session_state.action_points = 5 if mode == "easy" else 3

    if not is_decision_phase:
        progress_increase = 15 if mode == "hard" else 10
        st.session_state.disease_progress += progress_increase
        
        core_clues = ["犬吠样咳嗽", "夜间加重", "吸气性喉鸣", "三凹征"]
        if not any(c in st.session_state.unlocked_clues for c in core_clues):
            st.session_state.disease_progress += 5
            st.session_state.messages.append({"role": "assistant", "content": "（由于未触及核心线索，患儿病情在不知不觉中加重了...）"})
    
    st.session_state.respiratory_action_this_period = False
    st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress))
    
    if st.session_state.disease_progress >= 90:
        st.session_state.messages.append({"role": "assistant", "content": "🚨 监护仪发出刺耳的警报声！患儿出现严重呼吸衰竭征兆！"})
        st.session_state.game_over = True
    elif st.session_state.disease_progress >= 70:
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ 患儿出现明显三凹征，喉鸣音加重！情况紧急！"})

# ================= 7. 界面布局 =================

# 🌟 顶部状态徽章
st.markdown(f"""
<div class="status-bar">
    <div class="status-badge status-trust">
        <div class="status-label">❤️ 家长信任值</div>
        <div class="status-value">{st.session_state.trust_score} / 100</div>
    </div>
    <div class="status-badge status-action">
        <div class="status-label">⏳ 剩余行动点</div>
        <div class="status-value">{st.session_state.action_points} / {5 if mode == 'easy' else 3}</div>
    </div>
    <div class="status-badge status-disease">
        <div class="status-label">⚠️ 病情进展度</div>
        <div class="status-value">{st.session_state.disease_progress} / 100</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 🌟 三栏布局（电脑端）
if st.session_state.view_mode == "desktop":
    col_left, col_center, col_right = st.columns([1, 2.5, 1.2])
    
    # 左侧：生命体征
    with col_left:
        st.subheader("📈 实时生命体征")
        st.caption(f"当前时间：{SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[7])['time']}")
        dp = st.session_state.disease_progress
        
        if dp < 40:
            render_vital("SpO₂", "97%")
            render_vital("呼吸", "28 次/分")
            render_vital("心率", "120 次/分")
            render_vital("意识", "轻度烦躁")
        elif dp < 70:
            render_vital("SpO₂", "93%")
            render_vital("呼吸", "35 次/分")
            render_vital("心率", "140 次/分")
            render_vital("意识", "明显烦躁")
            if mode == "easy": st.warning("⚠️ 出现三凹征，需立即干预")
        elif dp < 90:
            render_vital("SpO₂", "88%")
            render_vital("呼吸", "45 次/分")
            render_vital("心率", "160 次/分")
            render_vital("意识", "发绀、烦躁")
            if mode == "easy": st.error("🚨 喉梗阻加重，随时可能呼吸衰竭")
            st.toast("🚨 生命体征危急！请立即处理！", icon="🚨")
        else:
            render_vital("SpO₂", "82%")
            render_vital("呼吸", "55 次/分")
            render_vital("心率", "180 次/分")
            render_vital("意识", "意识模糊")
            if mode == "easy": st.error("💀 极度危险！随时可能心跳骤停")
            st.toast("💀 患儿濒死！请立即抢救！", icon="💀")
        
        st.divider()
        if not st.session_state.game_over:
            if st.session_state.time_period in [1, 3]:
                if st.session_state.time_period == 3 and st.session_state.action_points <= 2 and not st.session_state.diagnosis_processed:
                    st.warning("⚠️ 问诊结束。请给出初步诊断：")
                    for opt_key, opt in DIAGNOSIS_OPTIONS.items():
                        if st.button(opt["label"], key=f"diag_{opt_key}", use_container_width=True):
                            st.session_state.diagnosis_processed = True
                            st.session_state.diagnosis_made = opt_key
                            st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + opt["disease_change"]))
                            if opt["is_correct"]:
                                st.session_state.score_diagnosis += opt["score"]
                                if st.session_state.first_act_clue and "一眼定乾坤" not in st.session_state.achievements:
                                    st.session_state.achievements.append("一眼定乾坤")
                            else:
                                st.session_state.penalty_log.append(f"诊断：误诊为{opt['label'][:6]}...")
                            st.session_state.messages.append({"role": "user", "content": f"【初步诊断】{opt['label']}"})
                            st.session_state.messages.append({"role": "assistant", "content": opt["reply"]})
                            st.rerun()
                elif st.session_state.action_points <= 2 and (st.session_state.time_period == 1 or st.session_state.diagnosis_processed):
                    if st.button("▶️ 进入下一幕", use
