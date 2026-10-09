import streamlit as st
import os
import re
import random
from openai import OpenAI

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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
    "A": {"label": "A. 吸气性喉鸣（Stridor）", "is_correct": True, "feedback": "✅ 正确！你听到了典型的吸气性喉鸣，这提示上气道梗阻，结合犬吠样咳嗽，高度支持急性喉炎！"},
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

# 🌟 错题本知识点映射表
ERROR_KNOWLEDGE = {
    "镇静剂": "❌ 错误操作：急性喉梗阻禁用镇静剂！镇静剂会抑制呼吸中枢，掩盖缺氧症状，极易导致呼吸骤停。正确做法：保持气道通畅、吸氧、雾化吸入肾上腺素。",
    "CT": "❌ 过度医疗：急性喉炎是临床诊断！不应等待CT或X线结果再处理，搬动患儿和长时间检查会加重喉水肿，延误抢救时机。",
    "拉肚子": "❌ 无效问诊：偏离主诉！急性喉炎的鉴别诊断核心在于呼吸系统，问诊应围绕咳嗽性质、呼吸困难和前驱症状。",
    "听诊：判断错误": "❌ 听诊错误：吸气性喉鸣（Stridor）提示上气道梗阻，常见于急性喉炎。呼气性哮鸣音（Wheezing）提示下气道梗阻，常见于哮喘。",
}

def get_local_reply(prompt, trust_score, period):
    # ... (此处保留上一版中完整的本地回复库，为了节省篇幅这里省略，实际使用时请复制之前的完整代码)
    return "医生，我太紧张了，您再说一遍好吗？"

def advance_period(is_decision_phase=False):
    st.session_state.time_period += 1
    
    if st.session_state.time_period == 4:
        st.session_state.action_points = 3
    else:
        st.session_state.action_points = 5

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

# ================= 4. 界面布局 =================
col_left, col_center, col_right = st.columns([1, 2.5, 1.2])

# ------------------ 左侧：状态栏（含生命体征面板） ------------------
with col_left:
    st.header("📋 急诊病历本")
    st.caption(f"当前时间：{SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[6])['time']}")
    st.metric(label="❤️ 家长信任值", value=f"{st.session_state.trust_score} / 100")
    st.progress(st.session_state.trust_score / 100)
    
    st.metric(label="⏳ 剩余行动点", value=f"{st.session_state.action_points} / 5")
    
    # 🌟 生命体征面板
    st.divider()
    st.subheader("📈 实时生命体征")
    dp = st.session_state.disease_progress
    if dp < 40:
        spo2, hr, rr, mental = "97%", "120次/分", "28次/分", "正常，轻度烦躁"
        c1, c2 = st.columns(2)
        c1.metric("SpO₂", spo2); c2.metric("心率", hr)
        c1.metric("呼吸", rr); c2.metric("意识", mental)
    elif dp < 70:
        spo2, hr, rr, mental = "93%", "140次/分", "35次/分", "明显烦躁"
        c1, c2 = st.columns(2)
        c1.metric("SpO₂", spo2); c2.metric("心率", hr)
        c1.metric("呼吸", rr); c2.metric("意识", mental)
        st.warning("⚠️ 出现三凹征，需立即干预")
    elif dp < 90:
        spo2, hr, rr, mental = "88%", "160次/分", "45次/分", "发绀、极度烦躁"
        c1, c2 = st.columns(2)
        c1.metric("SpO₂", spo2); c2.metric("心率", hr)
        c1.metric("呼吸", rr); c2.metric("意识", mental)
        st.error("🚨 喉梗阻加重，随时可能呼吸衰竭")
    else:
        spo2, hr, rr, mental = "82%", "180次/分", "55次/分", "意识模糊、濒死感"
        c1, c2 = st.columns(2)
        c1.metric("SpO₂", spo2); c2.metric("心率", hr)
        c1.metric("呼吸", rr); c2.metric("意识", mental)
        st.error("💀 极度危险！随时可能心跳骤停")
    
    st.divider()
    
    # 剧情推进按钮逻辑（保持原有逻辑）
    if not st.session_state.game_over:
        if st.session_state.time_period in [1, 3]:
            if st.session_state.time_period == 3 and st.session_state.action_points <= 2 and not st.session_state.diagnosis_processed:
                st.warning("⚠️ 问诊与查体阶段结束。请根据已有线索，给出你的初步诊断：")
                for opt_key, opt in DIAGNOSIS_OPTIONS.items():
                    if st.button(opt["label"], key=f"diag_{opt_key}", use_container_width=True):
                        st.session_state.diagnosis_made = opt_key
                        st.session_state.diagnosis_processed = True
                        st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + opt["disease_change"]))
                        if opt["is_correct"]:
                            st.session_state.score_diagnosis += opt["score"]
                        else:
                            st.session_state.penalty_log.append(f"第三幕诊断：误诊为{opt['label'][:6]}...")
                        st.session_state.messages.append({"role": "user", "content": f"【初步诊断】{opt['label']}"})
                        st.session_state.messages.append({"role": "assistant", "content": opt["reply"]})
                        st.rerun()
            
            elif st.session_state.action_points <= 2 and (st.session_state.time_period == 1 or st.session_state.diagnosis_processed):
                if st.button("▶️ 进入下一幕（已完成强制问诊）", use_container_width=True):
                    advance_period(is_decision_phase=False)
                    st.rerun()
            elif st.session_state.action_points > 2:
                st.caption(f"💡 强制问诊阶段：还需进行 {st.session_state.action_points - 2} 次问诊。")
        
        if st.session_state.time_period in [2, 5] and st.session_state.decision_made.get(st.session_state.time_period):
            if st.button("▶️ 继续剧情", use_container_width=True):
                advance_period(is_decision_phase=True)
                st.rerun()
        
        if st.session_state.time_period == 3 and st.session_state.diagnosis_processed and st.session_state.action_points <= 2:
            if st.button("🚨 突发事件！进入第四幕", use_container_width=True):
                st.session_state.time_period = 4
                st.session_state.action_points = 3 
                st.session_state.respiratory_action_this_period = False
                st.session_state.disease_progress = max(70, st.session_state.disease_progress + 10)
                st.session_state.messages.append({"role": "assistant", "content": "（患儿突然剧烈哭闹后，呼吸困难急剧加重，吸气时胸骨上窝明显凹陷，面色发绀）医生！她喘不上气了！您快救救她啊！"})
                st.rerun()
        
        if st.session_state.time_period == 4 and st.session_state.crisis_correct_count >= 2:
            if st.button("✅ 紧急处理完成，进入第五幕", use_container_width=True):
                st.session_state.score_emergency += min(25, st.session_state.crisis_correct_count * 10)
                st.session_state.time_period = 5
                st.session_state.action_points = 5
                st.session_state.respiratory_action_this_period = False
                st.session_state.messages.append({"role": "assistant", "content": "（经过你的紧急处理，患儿呼吸逐渐平稳，面色转红润。带教老师赶到现场...）"})
                st.rerun()
    
    st.divider()
    
    if not st.session_state.game_over and st.session_state.time_period >= 5:
        if st.button("📝 提交诊断，结束游戏", use_container_width=True):
            st.session_state.game_over = True
            st.rerun()
    elif not st.session_state.game_over:
        st.caption("💡 剧情推进到后期才会开放提交诊断")
    
    if st.button("🔄 重新开始游戏", use_container_width=True):
        restart_game()

# ------------------ 右侧：线索夹 ------------------
with col_right:
    st.header("🔍 线索夹")
    st.caption("通过问诊与查体解锁关键信息")
    st.divider()
    if len(st.session_state.unlocked_clues) == 0:
        st.info("暂无线索，快去问诊吧！")
    else:
        for clue in ALL_CLUES:
            if clue in st.session_state.unlocked_clues:
                st.success(f"✅ {clue}")
            else:
                st.text(f"🔒 未知线索")
    
    if st.session_state.penalty_log:
        st.divider()
        st.caption("📝 操作记录")
        for log in st.session_state.penalty_log:
            st.warning(f"⚠️ {log}")

# ------------------ 中间：主对话界面 ------------------
with col_center:
    if not st.session_state.game_over:
        current_scenario = SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[6])
        st.title(current_scenario["title"])
        st.info(f"**{current_scenario['task']}**\n\n{current_scenario['hint']}")
    else:
        st.title("🏥 午夜急诊 · 结案")

    with st.container(height=550):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # 听诊模式、决策模式、紧急处理、自由问诊模式（保持上一版的核心逻辑，此处省略以节省篇幅，请确保保留这些模块）

# ================= 游戏结算画面（含错题本） =================
if st.session_state.game_over:
    st.divider()
    st.header("🩺 带教老师复盘")
    
    # ... (保留上一版中计算总分、称号、AI复盘、最终数据的代码) ...

    # 🌟 错题本模块
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
