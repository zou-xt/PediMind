import streamlit as st
import os
import re
import random
from openai import OpenAI

# ================= 1. 配置区 =================
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

# ⚠️ 请替换为你刚刚测试成功的智谱AI API Key
API_KEY = st.secrets["ZHIPU_API_KEY"]
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

# 🌟 修复：删除了所有 label 里的 ✅ 和 ❌
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

# 🌟 本地回复生成器
def get_local_reply(prompt, trust_score, period):
    prompt_lower = prompt.lower()
    
    if trust_score >= 70:
        tone = "感激"
    elif trust_score >= 40:
        tone = "焦虑"
    else:
        tone = "急躁"
    
    intents = {
        "共情": {
            "keywords": ["别急", "送来得及时", "别怕", "我帮您", "冷静", "理解", "放心"],
            "replies": {
                "感激": ["呜呜呜谢谢医生，我尽量冷静……您快看看她吧。", "谢谢您理解，我真的是太害怕了，腿都软了。"],
                "焦虑": ["医生，您一定要救救她啊，我就这一个孩子……", "我尽量配合您，您说怎么办就怎么办。"],
                "急躁": ["希望您能理解我们做父母的心情！", "别光说好听的，快看看孩子到底怎么了！"]
            }
        },
        "冷漠": {
            "keywords": ["怎么才", "你怎么", "搞什么", "麻烦", "快点"],
            "replies": {
                "感激": ["对不起医生，我们太急了，您别生气……", "（带着哭腔）我们真的没办法了才半夜跑来的……"],
                "焦虑": ["（带着哭腔）我们已经尽快赶来了！您别凶我啊……", "您别这么说，孩子生病我们也不想啊，我都急死了！"],
                "急躁": ["你这是什么态度！我要投诉你！", "你懂不懂怎么当医生啊！孩子都这样了你还怪我！"]
            }
        },
        "咳嗽声音": {
            "keywords": ["小狗", "犬吠", "狗叫", "咳嗽声音", "什么样的咳"],
            "replies": {
                "感激": ["她咳起来'空空'的，像小狗叫一样，我录了视频您要看吗？"],
                "焦虑": ["就是那种'空空'的声音，像小狗叫！我从来没听过，吓死我了！", "咳嗽的声音特别吓人，'空空'的，嗓子也哑了，哭都哭不出声。"],
                "急躁": ["就是'空空'的声音！像小狗叫！你快点给她治啊！"]
            }
        },
        "发病时间": {
            "keywords": ["什么时候", "几点", "时间", "加重", "晚上", "半夜", "凌晨", "睡觉", "夜间"],
            "replies": {
                "感激": ["前天白天只是有点流鼻涕，凌晨1点多突然就开始咳了，咳得特别厉害。"],
                "焦虑": ["前天白天有点流鼻涕，半夜突然就咳醒了，大概凌晨1点多，之后就越来越重。", "就前天半夜，睡着睡着突然就咳醒了，一直没停过。"],
                "急躁": ["前天半夜开始的！这都问第三遍了，能不能先给孩子开药啊！"]
            }
        },
        "呼吸情况": {
            "keywords": ["吸气", "呼吸声", "喉鸣", "喘气声"],
            "replies": {
                "感激": ["吸气的时候有'吱吱'的声音，很吓人，您快听听。"],
                "焦虑": ["她吸气的时候有'吱吱'的声音，而且胸口这里凹进去一块，好吓人！", "呼吸特别费力，吸气的时候脖子下面都凹进去了。"],
                "急躁": ["喘不上气啊！吸气的时候'吱吱'响！你看不见吗？"]
            }
        },
        "白天情况": {
            "keywords": ["白天", "之前", "前几天", "感冒"],
            "replies": {
                "感激": ["白天就是流鼻涕，有点低烧，我以为普通感冒。"],
                "焦虑": ["白天就是有点流鼻涕，低烧，我给她喝了点感冒药。怎么晚上突然就成这样了？", "白天还好好的，就是有点感冒症状，怎么半夜突然就喘不上气了？"],
                "急躁": ["白天有点感冒！你到底能不能治？不能治我们转院！"]
            }
        },
        "查体": {
            "keywords": ["查体", "体格", "检查身体", "按压"],
            "replies": {
                "感激": ["（配合）您轻点……她胸口这里吸气的时候明显凹进去了。"],
                "焦虑": ["（配合）您轻点……她胸口这里吸气的时候明显凹进去了。", "好，我按住她，您快点查。"],
                "急躁": ["查查查，能不能先给点药啊！"]
            }
        }
    }
    
    for intent_name, intent_data in intents.items():
        if any(kw in prompt_lower for kw in intent_data["keywords"]):
            replies = intent_data["replies"].get(tone, intent_data["replies"]["焦虑"])
            available_replies = [r for r in replies if r not in st.session_state.local_reply_cache]
            if not available_replies:
                available_replies = replies
                st.session_state.local_reply_cache = []
            
            reply = random.choice(available_replies)
            st.session_state.local_reply_cache.append(reply)
            return reply
    
    fallbacks = {
        "感激": ["医生，她嗓子哑了，哭都哭不出声，拜托您了。", "我女儿就交给您了，有什么需要配合的您尽管说。"],
        "焦虑": ["医生，她嗓子哑了，哭都哭不出声，我该怎么办啊？", "您别光问，能不能先给她吸点氧啊？", "她越来越严重了，我真的很害怕……"],
        "急躁": ["别问了！先救人啊！", "你问这么多，到底什么时候能治啊？"]
    }
    available_fallbacks = [r for r in fallbacks.get(tone, fallbacks["焦虑"]) if r not in st.session_state.local_reply_cache]
    if not available_fallbacks:
        available_fallbacks = fallbacks.get(tone, fallbacks["焦虑"])
        st.session_state.local_reply_cache = []
    reply = random.choice(available_fallbacks)
    st.session_state.local_reply_cache.append(reply)
    return reply

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

# ------------------ 左侧：状态栏 ------------------
with col_left:
    st.header("📋 急诊病历本")
    st.caption(f"当前时间：{SCENARIO_DATA.get(st.session_state.time_period, SCENARIO_DATA[6])['time']}")
    st.metric(label="❤️ 家长信任值", value=f"{st.session_state.trust_score} / 100")
    st.progress(st.session_state.trust_score / 100)
    
    st.metric(label="⏳ 剩余行动点", value=f"{st.session_state.action_points} / 5")
    st.metric(label="⚠️ 病情进展度", value=f"{st.session_state.disease_progress} / 100")
    
    progress_value = min(1.0, max(0.0, st.session_state.disease_progress / 100))
    st.progress(progress_value)
    
    if st.session_state.disease_progress >= 90:
        st.error("🚨 极度危险！患儿出现呼吸衰竭征兆！")
    elif st.session_state.disease_progress >= 70:
        st.warning("⚠️ 患儿出现明显三凹征，喉鸣音加重！")
    elif st.session_state.disease_progress >= 50:
        st.info("ℹ️ 患儿病情正在进展，请尽快处理。")
    
    st.divider()
    
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
                            st.session_state.messages.append({"role": "user", "content": f"【初步诊断】{opt['label']}"})
                            st.session_state.messages.append({"role": "assistant", "content": opt["reply"]})
                            st.toast("✅ 诊断正确！病情缓解 -5", icon="📉")
                        else:
                            st.session_state.penalty_log.append(f"第三幕诊断：误诊为{opt['label'][:6]}...")
                            st.session_state.messages.append({"role": "user", "content": f"【初步诊断】{opt['label']}"})
                            st.session_state.messages.append({"role": "assistant", "content": opt["reply"]})
                            st.toast(f"❌ 误诊！病情急剧加重 +{opt['disease_change']}", icon="🚨")
                        st.rerun()
            
            elif st.session_state.action_points <= 2 and (st.session_state.time_period == 1 or st.session_state.diagnosis_processed):
                if st.button("▶️ 进入下一幕（已完成强制问诊）", use_container_width=True):
                    advance_period(is_decision_phase=False)
                    st.session_state.messages.append({"role": "assistant", "content": f"⏰ 时间推进到了 {SCENARIO_DATA[st.session_state.time_period]['time']}。"})
                    st.rerun()
            elif st.session_state.action_points > 2:
                st.caption(f"💡 强制问诊阶段：还需进行 {st.session_state.action_points - 2} 次问诊。")
        
        if st.session_state.time_period in [2, 5] and st.session_state.decision_made.get(st.session_state.time_period):
            if st.button("▶️ 继续剧情", use_container_width=True):
                advance_period(is_decision_phase=True)
                st.session_state.messages.append({"role": "assistant", "content": f"⏰ 时间推进到了 {SCENARIO_DATA[st.session_state.time_period]['time']}。"})
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

# ------------------ 中间：主对话界面（滚动区） ------------------
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

    # ================= 决策模式 =================
    if not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "decision":
        decision_key = None
        if st.session_state.time_period == 2:
            decision_key = 1
        elif st.session_state.time_period == 5:
            decision_key = 2
        
        if decision_key and not st.session_state.decision_made.get(st.session_state.time_period):
            decision = DECISIONS[decision_key]
            st.warning(f"⚠️ {decision['prompt']}")
            
            for opt_key, opt in decision["options"].items():
                if st.button(opt["label"], key=f"dec_{st.session_state.time_period}_{opt_key}", use_container_width=True):
                    st.session_state.decision_made[st.session_state.time_period] = opt_key
                    st.session_state.messages.append({"role": "user", "content": f"【决策】{opt['label']}"})
                    st.session_state.messages.append({"role": "assistant", "content": opt["reply"]})
                    
                    st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + opt["disease_change"]))
                    st.session_state.trust_score = max(0, min(100, st.session_state.trust_score + opt["trust_change"]))
                    
                    if opt["type"] != "correct":
                        type_map = {"invalid": "无效操作", "overuse": "过度医疗", "harmful": "有害操作"}
                        st.session_state.penalty_log.append(f"{SCENARIO_DATA[st.session_state.time_period]['time']}：{type_map[opt['type']]}")
                    
                    if st.session_state.disease_progress >= 100:
                        st.session_state.messages.append({"role": "assistant", "content": "（监护仪发出刺耳的警报声，患儿呼吸心跳骤停）......"})
                        st.session_state.game_over = True
                    
                    st.rerun()
        
        elif st.session_state.decision_made.get(st.session_state.time_period):
            choice_key = st.session_state.decision_made[st.session_state.time_period]
            choice = DECISIONS[decision_key]["options"][choice_key]
            
            if choice["type"] == "correct":
                st.success(f"✅ 你的选择：{choice['label']}\n\n{choice['reply']}")
            elif choice["type"] in ["invalid", "overuse"]:
                st.warning(f"⚠️ 你的选择：{choice['label']}\n\n{choice['reply']}")
            else:
                st.error(f"❌ 你的选择：{choice['label']}\n\n{choice['reply']}")
    
    # ================= 第四幕：紧急处理（动态病情） =================
    elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "crisis":
        st.warning("⚠️ 请从下列操作中选择紧急处理方案（每个操作消耗1个行动点，共3个行动点）：")
        
        if st.session_state.action_points > 0:
            for act_key, act in CRISIS_ACTIONS.items():
                if act_key not in st.session_state.crisis_actions:
                    if st.button(act["label"], key=f"crisis_{act_key}", use_container_width=True):
                        st.session_state.action_points -= 1
                        st.session_state.crisis_actions.append(act_key)
                        
                        if act["is_correct"]:
                            st.session_state.crisis_correct_count += 1
                            st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress - 15))
                            st.session_state.messages.append({"role": "user", "content": act["label"]})
                            st.session_state.messages.append({"role": "assistant", "content": act["feedback"] + "（患儿面色稍有缓解）"})
                            st.toast("✅ 正确操作！病情缓解 -15", icon="📉")
                        else:
                            st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + 25))
                            st.session_state.messages.append({"role": "user", "content": act["label"]})
                            st.session_state.messages.append({"role": "assistant", "content": act["feedback"] + "（患儿面色更加青紫！）"})
                            st.session_state.penalty_log.append(f"{SCENARIO_DATA[st.session_state.time_period]['time']}：有害操作")
                            st.toast("❌ 有害操作！病情急剧加重 +25", icon="🚨")
                        
                        if st.session_state.disease_progress >= 100:
                            st.session_state.game_over = True
                            st.rerun()
                        
                        st.rerun()
        
        if st.session_state.action_points <= 0 or len(st.session_state.crisis_actions) >= 3:
            st.divider()
            if st.session_state.crisis_correct_count >= 2:
                st.success(f"✅ 你做对了 {st.session_state.crisis_correct_count} 项正确操作！患儿症状开始缓解。请点击左侧『紧急处理完成』继续。")
            else:
                st.error(f"❌ 你只做对了 {st.session_state.crisis_correct_count} 项正确操作。患儿病情急剧恶化，触发 Bad Ending！")
                st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + 20))
                st.session_state.game_over = True
                st.rerun()
    
    # ================= 自由问诊模式（动态病情） =================
    elif not st.session_state.game_over and SCENARIO_DATA[st.session_state.time_period]["mode"] == "free":
        if st.session_state.time_period == 3 and st.session_state.action_points <= 2 and not st.session_state.diagnosis_processed:
            st.info("👉 请根据现有线索，在左侧【急诊病历本】中做出初步诊断。")
        else:
            if prompt := st.chat_input("请输入你的问诊、查体或检查操作..."):
                if st.session_state.action_points <= 0:
                    st.warning("本时段行动点已用完！请点击左侧『进入下一幕』。")
                else:
                    st.session_state.action_points -= 1
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    if st.session_state.action_points == 2:
                        st.toast("💡 已完成 3 次强制问诊！您还有 2 次可选问诊机会，或点击左侧『进入下一幕』。", icon="📋")
                    
                    trust_change = 0
                    disease_change = 5
                    clue = "无"
                    
                    empathy_kw = ["别急", "送来得及时", "别怕", "我帮您", "冷静", "理解", "放心"]
                    cold_kw = ["怎么才", "你怎么", "搞什么", "麻烦", "快点"]
                    if any(kw in prompt for kw in empathy_kw):
                        trust_change = 10
                        disease_change = 0
                        st.session_state.score_empathy = min(15, st.session_state.score_empathy + 5)
                    elif any(kw in prompt for kw in cold_kw):
                        trust_change = -15
                        disease_change = 10
                        st.session_state.score_empathy = max(0, st.session_state.score_empathy - 5)
                    
                    if any(k in prompt for k in ["小狗", "犬吠", "狗叫", "咳嗽声音", "什么样的咳"]):
                        clue = "犬吠样咳嗽"
                        disease_change = -5
                        st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8)
                    elif any(k in prompt for k in ["什么时候", "几点", "时间", "加重", "晚上", "半夜", "凌晨", "睡觉", "夜间"]):
                        clue = "夜间加重"
                        disease_change = -5
                        st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8)
                    elif "吸气" in prompt or "呼吸声" in prompt or "喉鸣" in prompt or "喘气声" in prompt:
                        clue = "吸气性喉鸣"
                        disease_change = -10
                        st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8)
                    elif "白天" in prompt or "之前" in prompt or "前几天" in prompt or "感冒" in prompt:
                        clue = "白天感冒史"
                        disease_change = 0
                        st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 6)
                    elif "查体" in prompt or "体格" in prompt or "检查身体" in prompt or "按压" in prompt:
                        clue = "三凹征"
                        disease_change = -15
                        st.session_state.score_inquiry = min(30, st.session_state.score_inquiry + 8)
                    
                    if "流口水" in prompt or "吞咽" in prompt or "会厌" in prompt:
                        st.session_state.score_diagnosis = min(30, st.session_state.score_diagnosis + 10)
                        disease_change = -5
                    elif "呛" in prompt or "异物" in prompt or "吃东西" in prompt:
                        st.session_state.score_diagnosis = min(30, st.session_state.score_diagnosis + 10)
                        disease_change = -5
                    elif "喘" in prompt or "哮喘" in prompt or "过敏" in prompt:
                        st.session_state.score_diagnosis = min(30, st.session_state.score_diagnosis + 10)
                        disease_change = -5
                    elif "拉肚子" in prompt or "皮疹" in prompt or "呕吐" in prompt:
                        disease_change = 15
                        st.toast("⚠️ 无效问诊！患儿病情加重！", icon="⚠️")
                    
                    st.session_state.disease_progress = max(0, min(100, st.session_state.disease_progress + disease_change))
                    
                    if disease_change > 0:
                        st.toast(f"⚠️ 病情进展 +{disease_change}！", icon="📈")
                    elif disease_change < 0:
                        st.toast(f"✅ 病情缓解 {disease_change}！", icon="📉")
                    
                    reply_text = None
                    try:
                        tone_prompt = "非常焦虑和自责" if st.session_state.trust_score < 40 else ("有些紧张但配合" if st.session_state.trust_score < 70 else "信任医生并感激")
                        ai_prompt = f"""
                        你正在扮演一个【2岁急性喉炎患儿的妈妈】，在医院急诊室。你当前的情绪状态是：{tone_prompt}。
                        
                        【剧本事实设定，必须严格遵守】
                        患儿小雨，2岁3个月。发病时间线是：前天白天只有轻微流鼻涕，凌晨1点半左右突然出现犬吠样咳嗽、声音嘶哑。之后症状在夜间进行性加重。
                        **绝对不能说“今天早上”、“昨天早上”等错误时间！必须回答“半夜”、“凌晨1点多”。**
                        
                        当前信任值：{st.session_state.trust_score}（低则急躁不配合，高则信任感激）。
                        医生刚刚说："{prompt}"
                        
                        请用不超过50字的口语化表达，直接回复医生的一句话。
                        要求：
                        1. 必须非常口语化，可以带哭腔、结巴、感叹词（如：呜呜、哎呀、天哪）。
                        2. 不要输出任何格式标记、表情符号或旁白，只输出纯文本的一句话。
                        """
                        ai_response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[{"role": "user", "content": ai_prompt}],
                            temperature=0.8,
                            timeout=3
                        )
                        reply_text = ai_response.choices[0].message.content.strip()
                    except Exception:
                        reply_text = get_local_reply(prompt, st.session_state.trust_score, st.session_state.time_period)
                    
                    if not reply_text:
                        reply_text = get_local_reply(prompt, st.session_state.trust_score, st.session_state.time_period)
                    
                    if any(k in reply_text for k in ["小狗", "犬吠", "狗叫"]):
                        if "犬吠样咳嗽" not in st.session_state.unlocked_clues:
                            clue = "犬吠样咳嗽"
                    
                    if "半夜" in reply_text or "凌晨" in reply_text or "睡着" in reply_text or "夜里" in reply_text:
                        if "夜间加重" not in st.session_state.unlocked_clues:
                            clue = "夜间加重"
                    
                    st.session_state.messages.append({"role": "assistant", "content": reply_text})
                    st.session_state.trust_score = max(0, min(100, st.session_state.trust_score + trust_change))
                    
                    if clue != "无" and clue in ALL_CLUES:
                        if clue not in st.session_state.unlocked_clues:
                            st.session_state.unlocked_clues.append(clue)
                            st.toast(f"🎉 解锁新线索：{clue}", icon="🔍")
                    
                    if st.session_state.disease_progress >= 100:
                        st.session_state.game_over = True
                    
                    st.rerun()

# ================= 游戏结算画面 =================
if st.session_state.game_over:
    st.divider()
    st.header("🩺 带教老师复盘")
    
    if st.session_state.final_score == 0:
        s_inquiry = st.session_state.score_inquiry
        s_diagnosis = st.session_state.score_diagnosis
        s_emergency = st.session_state.score_emergency
        s_empathy = min(15, st.session_state.score_empathy + (st.session_state.trust_score * 0.1))
        
        total = s_inquiry + s_diagnosis + s_emergency + s_empathy
        st.session_state.final_score = round(total, 1)
        
        if st.session_state.final_score >= 90:
            st.session_state.final_title = "🏆 儿科急诊之光"
        elif st.session_state.final_score >= 70:
            st.session_state.final_title = "🌟 有潜力的住院医"
        elif st.session_state.final_score >= 50:
            st.session_state.final_title = "📚 还需回炉重造"
        else:
            st.session_state.final_title = "😡 小雨妈妈已向医务科投诉"
    
    st.subheader("📊 评分维度")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("问诊完整性", f"{round(st.session_state.score_inquiry, 1)} / 30")
    with c2:
        st.metric("鉴别诊断", f"{round(st.session_state.score_diagnosis, 1)} / 30")
    with c3:
        st.metric("紧急处理", f"{round(st.session_state.score_emergency, 1)} / 25")
    with c4:
        s_empathy = min(15, st.session_state.score_empathy + (st.session_state.trust_score * 0.1))
        st.metric("医患沟通", f"{round(s_empathy, 1)} / 15")

    if st.session_state.final_evaluation is None:
        with st.spinner("带教老师正在仔细阅读你的问诊记录，评估你的表现..."):
            chat_history = "\n".join([f"{'医生' if m['role']=='user' else '家属'}: {m['content']}" for m in st.session_state.messages])
            penalty_text = "\n".join(st.session_state.penalty_log) if st.session_state.penalty_log else "无"
            
            eval_prompt = f"""
            你是资深儿科急诊带教老师。请点评这位医学生（医生）的表现。
            
            【最终状态】信任值:{st.session_state.trust_score} | 病情度:{st.session_state.disease_progress} | 收集线索:{st.session_state.unlocked_clues}
            
            【操作失误记录】
            {penalty_text}
            
            【问诊记录】
            {chat_history}
            
            【最终评分】
            问诊完整性:{st.session_state.score_inquiry}/30 | 鉴别诊断:{st.session_state.score_diagnosis}/30 | 紧急处理:{st.session_state.score_emergency}/25 | 医患沟通:{s_empathy}/15
            
            【要求】
            1.用老师口吻，先肯定优点再指出问题。
            2.重点点评：是否识别急性喉炎、是否排除了会厌炎/异物/哮喘、气道管理是否及时、是否有过度医疗或有害操作、医患沟通是否到位。
            3.结合得分情况给出具体的改进建议，300字左右。
            """
            
            try:
                eval_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": eval_prompt}],
                    temperature=0.7,
                    stream=True,
                    timeout=20
                )
                st.session_state.final_evaluation = st.write_stream(eval_response)
            except Exception as e:
                if st.session_state.final_score >= 90:
                    st.session_state.final_evaluation = "【系统自动评语】你表现非常出色！问诊全面，鉴别诊断精准，紧急处理果断，医患沟通到位。你准确识别了急性喉炎并排除了高危陷阱，是一名优秀的儿科急诊医生苗子。"
                elif st.session_state.final_score >= 70:
                    st.session_state.final_evaluation = "【系统自动评语】你表现良好，基本识别了急性喉炎，紧急处理方向正确。但在鉴别诊断或沟通细节上仍有提升空间。继续加油！"
                elif st.session_state.final_score >= 50:
                    st.session_state.final_evaluation = "【系统自动评语】你勉强及格。虽然最终患儿得到了救治，但你在问诊和鉴别诊断中出现了明显失误。建议回炉重造，重点复习急性喉炎的鉴别诊断和紧急处理流程。"
                else:
                    st.session_state.final_evaluation = "【系统自动评语】你的表现令人担忧。未能及时识别急性喉炎，操作中存在严重失误，患儿病情因此恶化。建议系统复习儿科急诊相关知识。"
    else:
        st.info(st.session_state.final_evaluation)
    
    st.divider()
    st.subheader("📊 最终结果")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("最终信任值", st.session_state.trust_score)
    with col2:
        st.metric("最终病情度", st.session_state.disease_progress)
    with col3:
        st.metric("线索收集数", f"{len(st.session_state.unlocked_clues)} / 5")
    
    st.divider()
    st.subheader("🏅 综合评价")
    st.metric("最终总分", f"{st.session_state.final_score} / 100")
    st.markdown(f"### {st.session_state.final_title}")
    
    if st.session_state.disease_progress >= 100:
        st.error("结局：Bad Ending。患儿因未及时处理喉梗阻，出现呼吸衰竭，被紧急气管插管。")
    elif st.session_state.disease_progress >= 90:
        st.error("结局：Bad Ending。患儿病情严重恶化，转入ICU。")
    elif st.session_state.trust_score < 40:
        st.warning("结局：家长因不信任你的沟通，抱着孩子转院了。")
    elif len(st.session_state.penalty_log) > 0:
        st.warning("结局：Neutral Ending。患儿最终好转，但你的操作存在明显失误，请复盘反思。")
    else:
        st.success("结局：Good Ending！你准确识别了急性喉炎，及时给予雾化吸入，患儿症状缓解。")
