import streamlit as st
import os
import re
import random
from openai import OpenAI
import plotly.graph_objects as go

# ================= 1. 配置区 =================
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

try:
    API_KEY = st.secrets["ZHIPU_API_KEY"]
except Exception:
    API_KEY = "sk-你的真实智谱AI密钥"  # 本地测试时替换

BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL_NAME = "glm-4-flash"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 2. 状态初始化 =================
st.set_page_config(page_title="急诊室疑云：2岁患儿的犬吠声", page_icon="🏥", layout="wide")

ALL_CLUES = ["犬吠样咳嗽", "夜间加重", "吸气性喉鸣", "白天感冒史", "三凹征"]

defaults = {
    "messages": [{"role": "assistant", "content": "医生！您快看看我家小雨！她半夜突然咳得像小狗叫一样，嗓子也哑了，我怎么哄都不行……白天就是有点流鼻涕，我给她喝了点感冒药，怎么会这样啊！"}],
    "trust_score": 40,
    "action_points": 5,
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
    "show_depression_img": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# 🌟 自定义生命体征渲染函数
def render_vital(label, value):
    st.markdown(
        f"""
        <div style="margin-bottom: 8px;">
            <div style="font-size: 12px; color: #888;">{label}</div>
            <div style="font-size: 16px; font-weight: 600; color: #333;">{value}</div>
        </div>
        """, unsafe_allow_html=True
    )

def restart_game():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ================= 3. 剧情配置区 =================
SCENARIO_DATA = {
    1: {"time": "凌晨 2:00", "title": "🎬 第一幕：急诊室初遇", "task": "任务目标：安抚家长情绪，通过问诊了解咳嗽的声音特征和发病时间规律。", "hint": "💡 提示：先共情安抚（如“别急，送来得及时”），再切入问诊。", "mode": "free"},
    2: {"time": "凌晨 2:30", "title": "🎬 第二幕：初步判断", "task": "任务目标：是否立刻给患儿开检查？请做出你的临床决策。", "hint": "💡 提示：考虑患儿当前病情和检查的优先级。", "mode": "decision"},
    3: {"time": "凌晨 2:45", "title": "🎬 第三幕：迷雾重重", "task": "任务目标：鉴别诊断。询问呼吸情况，并进行体格检查。注意排除会厌炎、异物、哮喘！", "hint": "💡 提示：重点关注“吸气性喉鸣”和“三凹征”，并针对性排除其他疾病。", "mode": "free"},
    4: {"time": "凌晨 3:00", "title": "🎬 第四幕：生死时速", "task": "突发事件！患儿出现吸气性呼吸困难加重，面色发绀，SpO₂降至88%！请立即处理！", "hint": "💡 提示：请在3个行动点内，从下方操作中选择正确的紧急处理组合。至少需要2项正确操作！", "mode": "crisis"},
    5: {"time": "凌晨 3:30", "title": "🎬 第五幕：带教老师介入", "task": "带教老师到场，询问你刚才的处理思路。你该如何回应？", "hint": "💡 提示：医患沟通与职业素养。", "mode": "decision"},
    6: {"time": "凌晨 4:00", "title": "🎬 第六幕：结案复盘", "task": "任务目标：提交你的最终诊断。", "hint": "💡 提示：带教老师正在看着你。", "mode": "end"}
}

DIAGNOSIS_OPTIONS = {
    "A": {"label": "A. 急性感染性喉炎（伴Ⅱ度喉梗阻）", "is_correct": True, "score": 15, "disease_change": -5, "reply": "✅ 诊断正确！你准确识别了犬吠样咳嗽、吸气性喉鸣和夜间加重三大特征。"},
    "B": {"label": "B. 急性会厌炎", "is_correct": False, "score": 0, "disease_change": 25, "reply": "❌ 误诊！患儿没有高热、流涎、吞咽困难，且存在典型的犬吠样咳嗽，不支持会厌炎。你的误判延误了抢救时机！"},
    "C": {"label": "C. 气道异物", "is_correct": False, "score": 0, "disease_change": 20, "reply": "❌ 误诊！患儿无突发剧烈呛咳史，且有前驱感冒症状，不支持气道异物。误诊导致你错过了最佳干预窗口！"},
    "D": {"label": "D. 支气管哮喘", "is_correct": False, "score": 0, "disease_change": 15, "reply": "❌ 误诊！患儿表现为吸气性呼吸困难（喉鸣），而非呼气性呼吸困难（哮鸣），且无过敏史。误诊导致病情进一步恶化！"}
}

AUSCULTATION_OPTIONS = {
    "A": {"label": "A. 吸气性喉鸣（Stridor）", "is_correct": True, "feedback": "✅ 正确！你听到了典型的吸气性喉鸣，这提示上气道梗阻。结合患儿‘犬吠样咳嗽’和‘夜间加重’的病史，你考虑最可能的诊断是什么？请继续收集线索，准备在第三幕给出你的初步诊断吧！"},
    "B": {"label": "B. 呼气性哮鸣音（Wheezing）", "is_correct": False, "feedback": "❌ 错误！你听到的是吸气性喉鸣，而不是呼气性哮鸣音。哮鸣音多见于哮喘或细支气管炎。"},
    "C": {"label": "C. 湿啰音（Crackles）", "is_correct": False, "feedback": "❌ 错误！湿啰音多见于肺炎或肺水肿，与本例上气道梗阻的听诊特征不符。"},
    "D": {"label": "D. 呼吸音正常", "is_correct": False, "feedback": "❌ 错误！患儿有明显的呼吸困难，听诊不可能完全正常。"}
}

CRISIS_ACTIONS = {
    "correct_1": {"label": "保持气道通畅：让患儿保持坐位/半坐位，避免哭闹加重喉水肿", "is_correct": True, "feedback": "（你让患儿保持坐位，呼吸稍有缓解）"},
    "correct_2": {"label": "氧疗：面罩吸氧", "is_correct": True, "feedback": "（面罩吸氧后，SpO₂开始缓慢回升）"},
    "correct_3": {"label": "雾化吸入：布地奈德+肾上腺素雾化（关键治疗）", "is_correct": True, "feedback": "（雾化吸入后，喉部水肿明显减轻，喉鸣音减弱）"},
    "correct_4": {"label": "静脉通路：开放静脉，准备糖皮质激素（地塞米松）", "is_correct": True, "feedback": "（静脉通路开放，为后续用药做好准备）"},
    "wrong_1": {"label": "强行按压患儿做咽喉部检查", "is_correct": False, "feedback": "（强行检查刺激喉部，患儿突发喉痉挛，喉鸣音消失，面色青紫！）"},
    "wrong_2": {"label": "使用镇静剂让患儿安静", "is_correct": False, "feedback": "（镇静剂使用后，患儿呼吸变浅变慢，血氧持续下降！）"},
    "wrong_3": {"label": "等待X线结果再处理", "is_correct": False, "feedback": "（等待影像结果的过程中，患儿病情急剧恶化！喉炎是临床诊断，不能等待影像！）"},
}

DECISIONS = {
    1: {
        "prompt": "患儿目前呼吸困难尚可，但声音嘶哑、夜间加重。你打算：",
        "options": {
            "A": {"label": "A. 立即进行床旁喉镜检查，明确喉部情况", "type": "correct", "disease_change": -5, "trust_change": 5, "reply": "（喉镜检查证实喉部黏膜充血水肿，符合喉炎表现）很好，你抓住了关键证据。"},
            "B": {"label": "B. 先观察，开点感冒药让家长回家", "type": "invalid", "disease_change": 25, "trust_change": -15, "reply": "（家长带着孩子离开，2小时后再次抱着孩子冲进来，此时患儿已出现三凹征）医生！她更严重了！"},
            "C": {"label": "C. 全套检查：血常规、CRP、胸部CT、心电图、心肌酶谱", "type": "overuse", "disease_change": 15, "trust_change": -10, "reply": "（折腾了1小时，患儿在检查过程中哭闹加剧，喉鸣音明显加重）医生，能不能先给孩子治治啊？"},
            "D": {"label": "D. 立即使用镇静剂让患儿安静下来配合检查", "type": "harmful", "disease_change": 35, "trust_change": -20, "reply": "（镇静剂使用后，患儿呼吸变浅变慢，血氧开始下降）医生！她怎么睡着了？叫不醒！"}
        }
    },
    2: {
        "prompt": "带教老师赶到，看了一眼监护仪，严肃地问你：刚才紧急处理时，你为什么要这样做？",
        "options": {
            "A": {"label": "A. 承认刚才有失误，详细复盘自己的判断过程，并说明后续改进方向", "type": "correct", "disease_change": 0, "trust_change": 10, "reply": "（带教老师点头）能反思就好。记住，气道急症不能等，处理顺序比检查更重要。"},
            "B": {"label": "B. 沉默不语，只是低头看着监护仪", "type": "invalid", "disease_change": 5, "trust_change": -5, "reply": "（带教老师皱眉）你连自己刚才做了什么都不敢面对吗？"},
            "C": {"label": "C. 把所有责任推给护士，说是护士操作不当", "type": "overuse", "disease_change": 5, "trust_change": -20, "reply": "（带教老师严肃）你作为医生，是团队的核心。推卸责任不是一个合格的医生该有的态度。"},
            "D": {"label": "D. 坚持认为自己处理完全正确，不承认任何问题", "type": "harmful", "disease_change": 10, "trust_change": -15, "reply": "（带教老师沉默片刻）你回去把急性喉炎的处理指南抄10遍，明天交给我。"}
        }
    }
}

ERROR_KNOWLEDGE = {
    "镇静剂": "❌ 错误操作：急性喉梗阻禁用镇静剂！镇静剂会抑制呼吸中枢，掩盖缺氧症状，极易导致呼吸骤停。正确做法：保持气道通畅、吸氧、雾化吸入肾上腺素。",
    "CT": "❌ 过度医疗：急性喉炎是临床诊断！不应等待CT或X线结果再处理，搬动患儿和长时间检查会加重喉水肿，延误抢救时机。",
    "拉肚子": "❌ 无效问诊：偏离主诉！急性喉炎的鉴别诊断核心在于呼吸系统，问诊应围绕咳嗽性质、呼吸困难和前驱症状。",
    "听诊：判断错误": "❌ 听诊错误：吸气性喉鸣（Stridor）提示上气道梗阻，常见于急性喉炎。呼气性哮鸣音（Wheezing）提示下气道梗阻，常见于哮喘。",
}

# ================= 4. 核心逻辑函数 =================
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

def advance_period(is_decision_phase=False):
    st.session_state.time_period += 1
    st.session_state.action_points = 3 if st.session_state.time_period == 4 else 5

    if not is_decision_phase:
        st.session_state.disease_progress += 10
        core_clues = ["犬吠样咳嗽", "夜间加重", "吸气性喉鸣", "三凹征"]
        if not any(c in st.session_state.unlocked_clues for c in core_clues):
            st.session_state.disease_progress += 5
            st.session_state.messages.append({"role": "assistant", "content": "（由于未触及核心线索，患儿病情在不知不觉中加重了...）"})
    
    st.session_state.respiratory_action_this_period = False
    st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress))
    
    if st.session_state.disease_progress >= 90:
        st.session_state.messages.append({"role": "assistant", "content": "🚨 监护仪发出刺耳的警报声！患儿出现严重呼吸衰竭征兆，口唇发绀，血氧持续下降！"})
        st.session_state.game_over = True
    elif st.session_state.disease_progress >= 70:
        st.session_state.messages.append({"role": "assistant", "content": "⚠️ 患儿出现明显三凹征，吸气时胸骨上窝、锁骨上窝、肋间隙凹陷，喉鸣音加重！情况紧急！"})

# ================= 5. 界面布局 =================
col_left, col_center, col_right = st.columns([1, 2.5, 1.2])

# ------------------ 左侧：生命体征与状态栏 ------------------
with col_left:
    st.header("📋 急诊病历本")
    st.caption(f"当前时间：{SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[6])['time']}")
    
    st.metric(label="❤️ 家长信任值", value=f"{st.session_state.trust_score} / 100")
    st.progress(st.session_state.trust_score / 100)
    st.metric(label="⏳ 剩余行动点", value=f"{st.session_state.action_points} / 5")
    
    st.divider()
    st.subheader("📈 实时生命体征")
    dp = st.session_state.disease_progress
    
    if dp < 40:
        c1, c2 = st.columns(2)
        with c1:
            render_vital("SpO₂", "97%")
            render_vital("呼吸", "28 次/分")
        with c2:
            render_vital("心率", "120 次/分")
            render_vital("意识", "轻度烦躁")
    elif dp < 70:
        c1, c2 = st.columns(2)
        with c1:
            render_vital("SpO₂", "93%")
            render_vital("呼吸", "35 次/分")
        with c2:
            render_vital("心率", "140 次/分")
            render_vital("意识", "明显烦躁")
        st.warning("⚠️ 出现三凹征，需立即干预")
    elif dp < 90:
        c1, c2 = st.columns(2)
        with c1:
            render_vital("SpO₂", "88%")
            render_vital("呼吸", "45 次/分")
        with c2:
            render_vital("心率", "160 次/分")
            render_vital("意识", "发绀、烦躁")
        st.error("🚨 喉梗阻加重，随时可能呼吸衰竭")
    else:
        c1, c2 = st.columns(2)
        with c1:
            render_vital("SpO₂", "82%")
            render_vital("呼吸", "55 次/分")
        with c2:
            render_vital("心率", "180 次/分")
            render_vital("意识", "意识模糊")
        st.error("💀 极度危险！随时可能心跳骤停")
    
    st.divider()
    
    if not st.session_state.game_over:
        if st.session_state.time_period in [1, 3]:
            if st.session_state.time_period == 3 and st.session_state.action_points <= 2 and not st.session_state.diagnosis_processed:
                st.warning("⚠️ 问诊结束。请给出初步诊断：")
                for opt_key, opt in DIAGNOSIS_OPTIONS.items():
                    if st.button(opt["label"], key=f"diag_{opt_key}", width="stretch"):
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
                if st.button("▶️ 进入下一幕", width="stretch"):
                    advance_period(is_decision_phase=False)
                    st.rerun()
            elif st.session_state.action_points > 2:
                st.caption(f"💡 强制问诊阶段：还需进行 {st.session_state.action_points - 2} 次问诊。")
        
        if st.session_state.time_period in [2, 5] and st.session_state.decision_made.get(st.session_state.time_period):
            if st.button("▶️ 继续剧情", width="stretch"):
                advance_period(is_decision_phase=True)
                st.rerun()
        
        if st.session_state.time_period == 3 and st.session_state.diagnosis_processed and st.session_state.action_points <= 2:
            if st.button("🚨 突发事件！进入第四幕", width="stretch"):
                st.session_state.time_period = 4
                st.session_state.action_points = 3 
                st.session_state.disease_progress = max(70, st.session_state.disease_progress + 10)
                st.session_state.messages.append({"role": "assistant", "content": "（患儿突然剧烈哭闹，呼吸困难急剧加重，面色发绀）医生！她喘不上气了！"})
                st.rerun()
        
        if st.session_state.time_period == 4 and st.session_state.crisis_correct_count >= 2:
            if st.button("✅ 进入第五幕", width="stretch"):
                st.session_state.score_emergency += min(25, st.session_state.crisis_correct_count * 10)
                st.session_state.time_period = 5
                st.session_state.action_points = 5
                st.session_state.messages.append({"role": "assistant", "content": "（经过处理，患儿呼吸逐渐平稳。带教老师赶到现场...）"})
                st.rerun()
    
    st.divider()
    if not st.session_state.game_over and st.session_state.time_period >= 5:
        if st.button("📝 提交诊断，结束游戏", width="stretch"):
            st.session_state.game_over = True
            st.rerun()
    elif not st.session_state.game_over:
        st.caption("💡 后期才会开放提交诊断")
    
    if st.button("🔄 重新开始游戏", width="stretch"):
        restart_game()

# ------------------ 右侧：线索夹与体征图库 ------------------
with col_right:
    st.header("🔍 线索夹")
    if len(st.session_state.unlocked_clues) == 0:
        st.info("暂无线索，快去问诊吧！")
    else:
        for clue in ALL_CLUES:
            if clue in st.session_state.unlocked_clues:
                st.success(f"✅ {clue}")
            else:
                st.text(f"🔒 未知线索")
                
    st.divider()
    st.subheader("🖼️ 体征图库")
    if not st.session_state.unlocked_clues:
        st.caption("解锁线索后，将在此显示对应体征图片。")
    else:
        if "犬吠样咳嗽" in st.session_state.unlocked_clues and os.path.exists("dog_cough.jpg"):
            st.image("dog_cough.jpg", caption="犬吠样咳嗽特征", width=300)
        if "吸气性喉鸣" in st.session_state.unlocked_clues and os.path.exists("stridor.jpg"):
            st.image("stridor.jpg", caption="吸气性喉鸣听诊波形", width=300)
        if "三凹征" in st.session_state.unlocked_clues and os.path.exists("three_depressions.jpg"):
            st.image("three_depressions.jpg", caption="患儿吸气时胸骨上窝、锁骨上窝及肋间隙明显凹陷", width=350)
    
    if st.session_state.penalty_log:
        st.divider()
        st.caption("📝 操作记录")
        for log in st.session_state.penalty_log:
            st.warning(f"⚠️ {log}")

# ------------------ 中间：主对话界面与查体 ------------------
with col_center:
    if not st.session_state.game_over:
        current_scenario = SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[6])
        st.title(current_scenario["title"])
        st.info(f"**{current_scenario['task']}**\n\n{current_scenario['hint']}")
    else:
        st.title("🏥 午夜急诊 · 结案")

    # 体格检查工具箱
    if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
        with st.expander("🩺 体格检查工具箱（点击展开，消耗行动点）", expanded=False):
            st.caption("💡 儿科急诊查体原则：先安抚，后检查；先救命，后诊病。")
            tab1, tab2, tab3, tab4 = st.tabs(["👁️ 视诊", "👂 听诊", "🖐️ 触诊", "🥁 叩诊"])
            
            with tab1:
                if st.button("观察呼吸系统", key="vis_resp"):
                    if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("vis_resp"):
                        st.session_state.action_points -= 1
                        st.session_state.physical_exam_done["vis_resp"] = True
                        st.session_state.messages.append({"role": "assistant", "content": "（你观察到：吸气时胸骨上窝、锁骨上窝明显凹陷，三凹征阳性！）"})
                        if "三凹征" not in st.session_state.unlocked_clues:
                            st.session_state.unlocked_clues.append("三凹征")
                        st.session_state.disease_progress = max(0, st.session_state.disease_progress - 5)
                        st.rerun()
            with tab2:
                if st.button("喉部听诊（音频判断）", key="aus_larynx"):
                    if st.session_state.action_points > 0 and not st.session_state.auscultation_completed:
                        st.session_state.action_points -= 1
                        st.session_state.auscultation_mode = True
                        st.rerun()
            with tab3:
                if st.button("腹部触诊", key="pal_abdomen"):
                    if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("pal_abdomen"):
                        st.session_state.action_points -= 1
                        st.session_state.physical_exam_done["pal_abdomen"] = True
                        st.session_state.messages.append({"role": "assistant", "content": "（患儿因缺氧哭闹剧烈，无法配合触诊。）"})
                        st.session_state.disease_progress = min(100, st.session_state.disease_progress + 5)
                        st.toast("⚠️ 查体不配合，病情 +5", icon="📈")
                        st.rerun()
            with tab4:
                if st.button("肺部叩诊", key="per_lung"):
                    if st.session_state.action_points > 0 and not st.session_state.physical_exam_done.get("per_lung"):
                        st.session_state.action_points -= 1
                        st.session_state.physical_exam_done["per_lung"] = True
                        st.session_state.messages.append({"role": "assistant", "content": "（患儿哭闹，无法配合叩诊。）"})
                        st.session_state.disease_progress = min(100, st.session_state.disease_progress + 5)
                        st.rerun()

    # 听诊模式 VS 正常对话模式
    if st.session_state.auscultation_mode:
        st.warning("🩺 你戴上听诊器，请仔细听诊患儿的呼吸音：")
        if os.path.exists("stridor.mp3"):
            st.audio("stridor.mp3", format="audio/mp3")
        else:
            st.info("🔇 未找到本地音频文件 'stridor.mp3'。请想象：吸气时出现高调、粗糙的'吱吱'声...")
        
        if os.path.exists("stridor.jpg"):
            st.image("stridor.jpg", caption="吸气性喉鸣音波形图", width=350)
            
        st.write("请判断你听到的是什么呼吸音：")
        for opt_key, opt in AUSCULTATION_OPTIONS.items():
            if st.button(opt["label"], key=f"aus_{opt_key}", width="stretch"):
                st.session_state.auscultation_mode = False
                st.session_state.auscultation_completed = True
                st.session_state.messages.append({"role": "user", "content": f"【听诊】{opt['label']}"})
                st.session_state.messages.append({"role": "assistant", "content": opt["feedback"]})
                if opt["is_correct"]:
                    st.session_state.score_diagnosis = min(30, st.session_state.score_diagnosis + 10)
                    st.session_state.disease_progress = max(0, st.session_state.disease_progress - 10)
                    if "吸气性喉鸣" not in st.session_state.unlocked_clues:
                        st.session_state.unlocked_clues.append("吸气性喉鸣")
                    st.toast("✅ 听诊正确！解锁线索【吸气性喉鸣】，病情缓解 -10", icon="📉")
                else:
                    st.session_state.penalty_log.append("听诊：判断错误")
                    st.session_state.disease_progress = min(100, st.session_state.disease_progress + 10)
                    st.toast("❌ 听诊错误！病情加重 +10", icon="🚨")
                st.rerun()
                
    else:
        with st.container(height=450):
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        # 决策模式按钮
        if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "decision":
            decision_key = 1 if st.session_state.time_period == 2 else 2
            if not st.session_state.decision_made.get(st.session_state.time_period):
                decision = DECISIONS[decision_key]
                st.warning(f"⚠️ {decision['prompt']}")
                for opt_key, opt in decision["options"].items():
                    if st.button(opt["label"], key=f"dec_{st.session_state.time_period}_{opt_key}", width="stretch"):
                        st.session_state.decision_made[st.session_state.time_period] = opt_key
                        st.session_state.messages.append({"role": "user", "content": f"【决策】{opt['label']}"})
                        st.session_state.messages.append({"role": "assistant", "content": opt["reply"]})
                        st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + opt["disease_change"]))
                        st.session_state.trust_score = max(0, min(100, st.session_state.trust_score + opt["trust_change"]))
                        if opt["type"] != "correct":
                            st.session_state.penalty_log.append(f"决策失误：{opt['label'][:8]}...")
                        st.rerun()

        # 紧急处理模式按钮
        elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "crisis":
            st.warning("⚠️ 请从下列操作中选择紧急处理方案（消耗行动点）：")
            if st.session_state.action_points > 0:
                for act_key, act in CRISIS_ACTIONS.items():
                    if act_key not in st.session_state.crisis_actions:
                        if st.button(act["label"], key=f"crisis_{act_key}", width="stretch"):
                            st.session_state.action_points -= 1
                            st.session_state.crisis_actions.append(act_key)
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
                if st.session_state.crisis_correct_count >= 2:
                    st.success(f"✅ 你做对了 {st.session_state.crisis_correct_count} 项正确操作！请点击左侧『进入第五幕』。")
                else:
                    st.error(f"❌ 正确操作不足2项。患儿病情急剧恶化，触发 Bad Ending！")
                    st.session_state.disease_progress = min(100, st.session_state.disease_progress + 20)
                    st.session_state.game_over = True
                    st.rerun()

        # 自由问诊输入框
        elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
            if st.session_state.time_period == 3 and st.session_state.action_points <= 2 and not st.session_state.diagnosis_processed:
                st.info("👉 请根据现有线索，在左侧做出初步诊断。")
            else:
                if prompt := st.chat_input("请输入你的问诊、查体或检查操作..."):
                    if st.session_state.action_points <= 0:
                        st.warning("行动点已用完！请点击左侧『进入下一幕』。")
                    else:
                        st.session_state.action_points -= 1
                        st.session_state.messages.append({"role": "user", "content": prompt})
                        
                        # 关键词拦截听诊
                        if any(k in prompt for k in ["听诊", "听呼吸", "听肺"]):
                            if not st.session_state.auscultation_completed:
                                st.session_state.auscultation_mode = True
                                st.rerun()
                            else:
                                st.session_state.messages.append({"role": "assistant", "content": "（你已经听过了，患儿现在很烦躁，不宜反复听诊。）"})
                                st.rerun()
                        
                        # 成就追踪
                        if st.session_state.time_period == 1 and any(k in prompt for k in ["小狗", "犬吠", "狗叫", "咳嗽声音"]):
                            st.session_state.first_act_clue = True
                        if any(k in prompt for k in ["流口水", "吞咽", "会厌"]):
                            if "会厌炎" not in st.session_state.excluded_diseases: st.session_state.excluded_diseases.append("会厌炎")
                        if any(k in prompt for k in ["呛", "异物", "吃东西"]):
                            if "异物" not in st.session_state.excluded_diseases: st.session_state.excluded_diseases.append("异物")
                        if any(k in prompt for k in ["喘", "哮喘", "过敏"]):
                            if "哮喘" not in st.session_state.excluded_diseases: st.session_state.excluded_diseases.append("哮喘")
                        if len(st.session_state.excluded_diseases) == 3:
                            if "千金难买早知道" not in st.session_state.achievements:
                                st.session_state.achievements.append("千金难买早知道")
                                st.toast("🎉 解锁成就：千金难买早知道！", icon="🏅")

                        # 动态病情判断
                        trust_change = 0
                        disease_change = 5
                        clue = "无"
                        if any(kw in prompt for kw in ["别急", "送来得及时", "别怕", "我帮您", "冷静", "理解", "放心"]):
                            trust_change = 10; disease_change = 0
                            st.session_state.score_empathy = min(15, st.session_state.score_empathy + 5)
                        elif any(kw in prompt for kw in ["怎么才", "你怎么", "搞什么", "麻烦", "快点"]):
                            trust_change = -15; disease_change = 10
                        
                        if any(k in prompt for k in ["小狗", "犬吠", "狗叫", "咳嗽声音", "什么样的咳"]):
                            clue = "犬吠样咳嗽"; disease_change = -5
                            st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8)
                            st.session_state.show_dog_cough_img = True
                        elif any(k in prompt for k in ["什么时候", "几点", "时间", "加重", "晚上", "半夜", "凌晨"]):
                            clue = "夜间加重"; disease_change = -5
                            st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8)
                        elif any(k in prompt for k in ["吸气", "呼吸声", "喉鸣", "喘气声"]):
                            clue = "吸气性喉鸣"; disease_change = -10
                            st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8)
                            st.session_state.show_stridor_img = True
                        elif any(k in prompt for k in ["白天", "之前", "前几天", "感冒"]):
                            clue = "白天感冒史"; disease_change = 0
                            st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 6)
                        elif any(k in prompt for k in ["拉肚子", "皮疹", "呕吐"]):
                            disease_change = 15
                            st.toast("⚠️ 无效问诊！病情加重！", icon="⚠️")
                        
                        st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + disease_change))
                        
                        # 调用 AI
                        reply_text = None
                        try:
                            tone_prompt = "非常焦虑和自责" if st.session_state.trust_score < 40 else ("有些紧张但配合" if st.session_state.trust_score < 70 else "信任医生并感激")
                            ai_prompt = f"你是2岁急性喉炎患儿的妈妈，在急诊室跟医生对话。情绪:{tone_prompt}。事实:前天白天流鼻涕，凌晨1点半突发犬吠样咳嗽。铁律：1.对医生说话，禁止对宝宝自言自语。2.句子必须完整，逻辑通顺。3.回复在50字以内。医生问：'{prompt}'。请直接回答医生："
                            ai_response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": ai_prompt}], temperature=0.4, timeout=3)
                            reply_text = ai_response.choices[0].message.content.strip()
                        except Exception:
                            reply_text = get_local_reply(prompt, st.session_state.trust_score, st.session_state.time_period)
                        
                        if not reply_text: reply_text = get_local_reply(prompt, st.session_state.trust_score, st.session_state.time_period)
                        
                        # 强制防智障过滤
                        bad_phrases = ["宝宝", "妈妈对不起", "咱们要坚强", "乖", "妈妈在", "吓坏妈妈", "那么厉害", "谢谢你啊"]
                        if any(k in reply_text for k in bad_phrases) or len(reply_text) < 5:
                            reply_text = "医生，呜呜，她咳得喘不上气，您快救救她吧！"
                        if len(reply_text) > 60:
                            reply_text = reply_text[:60] + "..."
                        
                        if any(k in reply_text for k in ["小狗", "犬吠", "狗叫"]):
                            if "犬吠样咳嗽" not in st.session_state.unlocked_clues: clue = "犬吠样咳嗽"
                        if any(k in reply_text for k in ["半夜", "凌晨", "睡着", "夜里"]):
                            if "夜间加重" not in st.session_state.unlocked_clues: clue = "夜间加重"
                        
                        st.session_state.messages.append({"role": "assistant", "content": reply_text})
                        st.session_state.trust_score = max(0, min(100, st.session_state.trust_score + trust_change))
                        if clue != "无" and clue in ALL_CLUES:
                            if clue not in st.session_state.unlocked_clues:
                                st.session_state.unlocked_clues.append(clue)
                                st.toast(f"🎉 解锁新线索：{clue}", icon="🔍")
                        
                        if st.session_state.disease_progress >= 100: 
                            st.session_state.game_over = True
                            if "反面教材" not in st.session_state.achievements:
                                st.session_state.achievements.append("反面教材")
                        st.rerun()

# ================= 6. 游戏结算与复盘 =================
if st.session_state.game_over:
    st.divider()
    st.header("🩺 带教老师复盘")
    
    if st.session_state.disease_progress >= 100:
        if "反面教材" not in st.session_state.achievements:
            st.session_state.achievements.append("反面教材")
    
    if st.session_state.final_score == 0:
        s_inquiry = st.session_state.score_inquiry
        s_diagnosis = st.session_state.score_diagnosis
        s_emergency = st.session_state.score_emergency
        s_empathy = min(15, st.session_state.score_empathy + (st.session_state.trust_score * 0.1))
        total = s_inquiry + s_diagnosis + s_emergency + s_empathy
        st.session_state.final_score = round(total, 1)
        
        if st.session_state.final_score >= 90: st.session_state.final_title = "🏆 儿科急诊之光"
        elif st.session_state.final_score >= 70: st.session_state.final_title = "🌟 有潜力的住院医"
        elif st.session_state.final_score >= 50: st.session_state.final_title = "📚 还需回炉重造"
        else: st.session_state.final_title = "😡 小雨妈妈已向医务科投诉"
    
    # 1. AI 带教老师点评
    if st.session_state.final_evaluation is None:
        with st.spinner("带教老师正在复盘..."):
            chat_history = "\n".join([f"{'医生' if m['role']=='user' else '家属'}: {m['content']}" for m in st.session_state.messages])
            penalty_text = "\n".join(st.session_state.penalty_log) if st.session_state.penalty_log else "无"
            eval_prompt = f"你是儿科急诊带教老师。请点评这位医学生的表现。\n【状态】信任值:{st.session_state.trust_score} | 病情度:{st.session_state.disease_progress} | 线索:{st.session_state.unlocked_clues}\n【失误】{penalty_text}\n【记录】{chat_history}\n【要求】先肯定优点再指出问题，结合得分给出改进建议，300字左右。"
            try:
                eval_response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": eval_prompt}], temperature=0.7, stream=True, timeout=20)
                st.session_state.final_evaluation = st.write_stream(eval_response)
            except Exception:
                st.session_state.final_evaluation = "【系统自动评语】你完成了本次急救演练。请在错题本中复习失误点，重点掌握急性喉炎的鉴别诊断和紧急处理流程。"
    else:
        st.info(st.session_state.final_evaluation)
    
    # 2. 个人能力雷达图
    st.divider()
    st.subheader("📊 个人能力雷达图")
    s_empathy_final = min(15, st.session_state.score_empathy + (st.session_state.trust_score * 0.1))
    scores = [st.session_state.score_inquiry, st.session_state.score_diagnosis, st.session_state.score_emergency, s_empathy_final]
    max_scores = [30, 30, 25, 15]
    normalized_scores = [(s / m) * 100 for s, m in zip(scores, max_scores)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_scores,
        theta=['问诊完整性 (30)', '鉴别诊断 (30)', '紧急处理 (25)', '医患沟通 (15)'],
        fill='toself',
        name='个人能力'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        height=400,
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig, width="stretch")

    # 3. 动态诊断树复盘
    st.divider()
    st.subheader("🌳 动态诊断树复盘")
    has_laryngitis = st.session_state.diagnosis_made == "A"
    excluded_epiglottitis = "会厌炎" in st.session_state.excluded_diseases
    excluded_foreign_body = "异物" in st.session_state.excluded_diseases
    excluded_asthma = "哮喘" in st.session_state.excluded_diseases
    
    tree_text = "呼吸困难\n"
    tree_text += "├── 吸气性（喉鸣）\n"
    tree_text += f"│   ├── 急性喉炎 {'✅ (已确诊，犬吠样咳嗽/三凹征阳性)' if has_laryngitis else '❌ (未能确诊，延误治疗)'}\n"
    tree_text += f"│   ├── 会厌炎 {'❌ (你问了流口水，排除)' if excluded_epiglottitis else '⚠️ (未评估风险)'}\n"
    tree_text += f"│   └── 异物 {'❌ (你问了呛咳史，排除)' if excluded_foreign_body else '⚠️ (未评估风险)'}\n"
    tree_text += "└── 呼气性（哮鸣）\n"
    tree_text += f"    └── 哮喘 {'❌ (你问了过敏史，排除)' if excluded_asthma else '⚠️ (未评估风险)'}\n"
    st.code(tree_text, language="text")

    # 4. 错题本
    st.divider()
    st.subheader("📝 错题本与知识点复盘")
    if not st.session_state.penalty_log:
        st.success("🎉 完美！你没有任何失误记录，展现出了扎实的临床基本功！")
    else:
        for log in st.session_state.penalty_log:
            st.warning(f"📌 {log}")
            matched = False
            for key, knowledge in ERROR_KNOWLEDGE.items():
                if key in log:
                    st.info(knowledge)
                    matched = True
                    break
            if not matched:
                st.info("💡 临床提示：请回顾该操作是否违背了急诊“先救命、后治病”的原则。")

    # 5. 成就展示
    st.divider()
    st.subheader("🏆 隐藏成就")
    if not st.session_state.achievements:
        st.write("暂无成就，再接再厉！")
    else:
        for ach in st.session_state.achievements:
            if ach == "一眼定乾坤":
                st.markdown("🥇 **一眼定乾坤**：第一幕就问出关键症状并成功确诊！")
            elif ach == "千金难买早知道":
                st.markdown("🥇 **千金难买早知道**：排除了所有高危鉴别诊断（会厌炎、异物、哮喘）！")
            elif ach == "反面教材":
                st.markdown("🥇 **反面教材**：患儿病情达到极度危险状态，请吸取教训！")

    st.divider()
    st.subheader("🏅 综合评价")
    st.metric("最终总分", f"{st.session_state.final_score} / 100")
    st.markdown(f"### {st.session_state.final_title}")
    
    if st.session_state.disease_progress >= 100:
        st.error("结局：Bad Ending。患儿因未及时处理喉梗阻，出现呼吸衰竭，被紧急气管插管。")
    elif st.session_state.trust_score < 40:
        st.warning("结局：家长因不信任你的沟通，抱着孩子转院了。")
    elif len(st.session_state.penalty_log) > 0:
        st.warning("结局：Neutral Ending。患儿最终好转，但你的操作存在明显失误，请复盘反思。")
    else:
        st.success("结局：Good Ending！你准确识别了急性喉炎，及时给予雾化吸入，患儿症状缓解。")
