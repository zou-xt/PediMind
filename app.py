import streamlit as st
import time
import random
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Dict, List, Optional

# ===================== 全局配置（不改动游戏内容） =====================
st.set_page_config(
    page_title="医学病例模拟游戏",
    layout="wide",
    initial_sidebar_state="auto"
)

# 成就配置（原样保留）
ACHIEVEMENTS = {
    "first_case": {"name": "初诊医师", "desc": "完成第一例病例", "icon": "🩺"},
    "perfect": {"name": "满分诊疗", "desc": "单病例获得满分", "icon": "⭐"},
    "speed": {"name": "急救先锋", "desc": "90秒内完成病例", "icon": "⚡"},
    "all_cases": {"name": "全科能手", "desc": "通关全部病例", "icon": "🏆"}
}

# 病例库（原样保留，急性喉炎等题目不变）
CASE_LIBRARY = [
    {
        "case_id": "case_01",
        "title": "急性喉炎",
        "info": "患儿，3岁，夜间突发犬吠样咳嗽，吸气性喉鸣，声音嘶哑，轻度吸气性呼吸困难。无发热，既往体健。",
        "questions": [
            {
                "q": "首要评估重点？",
                "options": ["肺部啰音", "气道通畅度", "肝脾大小", "尿量"],
                "ans": 1,
                "hint": "急性喉炎最危险的是喉头水肿堵塞气道"
            },
            {
                "q": "首选紧急处理？",
                "options": ["口服抗生素", "布地奈德雾化+吸氧、监护", "大量补液", "立即胸片"],
                "ans": 1,
                "hint": "糖皮质激素雾化减轻喉头水肿，优先保障通气"
            },
            {
                "q": "出现三度喉梗阻下一步？",
                "options": ["继续雾化观察", "准备气管插管/气管切开", "口服退烧药", "转普通门诊"],
                "ans": 1,
                "hint": "三度喉梗阻有窒息风险，需做好气道急救准备"
            }
        ],
        "ecg_enable": True
    }
]

# 心电图波形参数
ECG_BASE_HZ = 1.0
ECG_AMPLITUDE = 1.2

@dataclass
class PlayerRecord:
    player_id: str
    score: int = 0
    finished_cases: List[str] = None
    achievements: List[str] = None
    start_time: Optional[float] = None

    def __post_init__(self):
        if self.finished_cases is None:
            self.finished_cases = []
        if self.achievements is None:
            self.achievements = []


def init_session_state():
    """【修复报错】兜底初始化所有session变量，防止key不存在报错"""
    default_keys = {
        "player_map": dict(),
        "current_player": None,
        "current_case": None,
        "q_index": 0,
        "case_score": 0,
        "case_start_ts": None,
        "ecg_running": False,
        "ecg_phase": 0,
        "show_hint": False
    }
    for k, v in default_keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # 兼容旧数据，防止类型异常
    if not isinstance(st.session_state["player_map"], dict):
        st.session_state["player_map"] = {}


def get_or_create_player(pid: str) -> PlayerRecord:
    """【修复报错】玩家读取，异常时自动重建玩家记录"""
    try:
        if pid not in st.session_state["player_map"]:
            st.session_state["player_map"][pid] = PlayerRecord(player_id=pid)
        rec = st.session_state["player_map"][pid]
        if not isinstance(rec, PlayerRecord):
            rec = PlayerRecord(player_id=pid)
            st.session_state["player_map"][pid] = rec
        return rec
    except Exception:
        new_rec = PlayerRecord(player_id=pid)
        st.session_state["player_map"][pid] = new_rec
        return new_rec


def generate_ecg_wave(phase: float, heart_rate: int = 110) -> np.ndarray:
    """
    生成模拟心电图波形，原有波形逻辑完全不变
    增加数值边界保护，防止数值溢出报错
    """
    try:
        t = np.linspace(0, 2 * np.pi, 120)
        base_t = t + phase

        # P波
        p_wave = 0.25 * np.sin(base_t) * np.where((base_t > 0) & (base_t < np.pi/3), 1, 0)
        # QRS波群
        qrs = np.zeros_like(base_t)
        qrs_mask = (base_t > np.pi/3) & (base_t < np.pi/2)
        qrs[qrs_mask] = -0.6 * np.sin(8 * base_t[qrs_mask])
        qrs_mask2 = (base_t > np.pi/2) & (base_t < 2*np.pi/3)
        qrs[qrs_mask2] = 1.1 * np.sin(8 * base_t[qrs_mask2])
        # T波
        t_wave = 0.35 * np.sin(base_t) * np.where((base_t > 2*np.pi/3) & (base_t < np.pi), 1, 0)

        wave = p_wave + qrs + t_wave
        # 限制幅值范围，避免绘图异常
        wave = np.clip(wave, -1.5, 1.5)
        return wave
    except Exception:
        # 异常兜底：返回平直基线
        return np.zeros(120)


def draw_ecg_component():
    """
    心电图展示组件，原有渲染逻辑不变
    增加状态判断兜底，防止未初始化时报错
    """
    try:
        if "ecg_running" not in st.session_state or "ecg_phase" not in st.session_state:
            st.session_state["ecg_running"] = False
            st.session_state["ecg_phase"] = 0

        if not st.session_state["ecg_running"]:
            st.info("🫀 心电图监测：未启动")
            return

        wave_data = generate_ecg_wave(st.session_state["ecg_phase"])
        x_axis = list(range(len(wave_data)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=wave_data,
            line=dict(color="#ff3333", width=2),
            mode="lines"
        ))
        fig.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[-1.6, 1.6], showticklabels=False),
            xaxis=dict(showticklabels=False),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        # 更新相位
        st.session_state["ecg_phase"] += 0.15
        if st.session_state["ecg_phase"] > 2 * np.pi:
            st.session_state["ecg_phase"] = 0
    except Exception as e:
        st.warning(f"心电图渲染临时异常：{str(e)}")


def check_achievements(player: PlayerRecord, case_obj: dict, case_score: int, total_q: int):
    """
    成就判定逻辑【完全保留原有规则】
    增加空值校验，防止读取病例、时间为空时报错
    """
    try:
        if "first_case" not in player.achievements and len(player.finished_cases) >= 1:
            player.achievements.append("first_case")
            st.success("🏅 解锁成就：初诊医师")

        if "perfect" not in player.achievements and case_score == total_q:
            player.achievements.append("perfect")
            st.success("🏅 解锁成就：满分诊疗")

        start_ts = st.session_state.get("case_start_ts")
        if "speed" not in player.achievements and start_ts is not None:
            cost = time.time() - start_ts
            if cost <= 90:
                player.achievements.append("speed")
                st.success("🏅 解锁成就：急救先锋")

        all_case_ids = [c["case_id"] for c in CASE_LIBRARY]
        if "all_cases" not in player.achievements and set(player.finished_cases) == set(all_case_ids):
            player.achievements.append("all_cases")
            st.success("🏅 解锁成就：全科能手")
    except Exception:
        st.warning("成就检测异常，不影响继续游戏")


def render_case_page(player: PlayerRecord):
    """
    病例答题页面，原有答题逻辑完全不变
    增加大量空值、下标边界捕获，防止索引越界、key缺失报错
    """
    try:
        case = st.session_state.get("current_case")
        q_idx = st.session_state.get("q_index", 0)
        case_score = st.session_state.get("case_score", 0)

        # 兜底校验：病例为空就退回首页
        if case is None:
            st.warning("病例信息异常，返回首页")
            st.session_state["current_case"] = None
            st.session_state["q_index"] = 0
            st.rerun()
            return

        questions = case.get("questions", [])
        # 题目列表为空保护
        if not isinstance(questions, list) or len(questions) == 0:
            st.warning("该病例题目数据异常")
            st.session_state["current_case"] = None
            st.rerun()
            return

        # 下标越界兜底
        if q_idx < 0 or q_idx >= len(questions):
            st.session_state["q_index"] = 0
            st.rerun()
            return

        current_q = questions[q_idx]
        st.subheader(f"📋 {case.get('title', '未知病例')}")
        st.info(case.get("info", ""))
        st.divider()

        # 心电图模块渲染
        if case.get("ecg_enable", False):
            draw_ecg_component()
            st.divider()

        st.markdown(f"### 问题 {q_idx + 1}/{len(questions)}")
        st.write(current_q.get("q", ""))
        options = current_q.get("options", [])
        ans_idx = current_q.get("ans", 0)
        hint_text = current_q.get("hint", "")

        # 选项异常兜底
        if not isinstance(options, list) or len(options) == 0:
            st.error("选项数据缺失")
            return

        selected = st.radio("请选择你的答案：", options, key=f"q_{q_idx}_{case['case_id']}")
        selected_idx = options.index(selected)

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            submit_btn = st.button("提交答案")
        with col2:
            show_hint_btn = st.button("查看提示")
        with col3:
            exit_case_btn = st.button("退出本病例")

        if show_hint_btn:
            st.session_state["show_hint"] = True

        if st.session_state.get("show_hint", False):
            st.caption(f"💡提示：{hint_text}")

        if exit_case_btn:
            st.session_state["current_case"] = None
            st.session_state["q_index"] = 0
            st.session_state["case_score"] = 0
            st.session_state["ecg_running"] = False
            st.session_state["show_hint"] = False
            st.rerun()
            return

        if submit_btn:
            if selected_idx == ans_idx:
                st.success("✅ 回答正确！")
                case_score += 1
                st.session_state["case_score"] = case_score
            else:
                st.error(f"❌ 回答错误。正确答案：{options[ans_idx]}")
                st.caption(f"💡 {hint_text}")

            time.sleep(1.2)
            # 下一题
            next_q = q_idx + 1
            if next_q >= len(questions):
                # 病例完成
                case_id = case.get("case_id")
                if case_id not in player.finished_cases:
                    player.finished_cases.append(case_id)
                # 成就校验
                check_achievements(player, case, case_score, len(questions))
                player.score += case_score
                st.session_state["player_map"][player.player_id] = player

                st.success(f"🎉 本病例完成！得分：{case_score}/{len(questions)}")
                st.session_state["ecg_running"] = False
                st.session_state["current_case"] = None
                st.session_state["q_index"] = 0
                st.session_state["case_score"] = 0
                st.session_state["show_hint"] = False
                time.sleep(2)
                st.rerun()
            else:
                st.session_state["q_index"] = next_q
                st.session_state["show_hint"] = False
                st.rerun()
    except IndexError:
        st.warning("题目索引异常，重置病例")
        st.session_state["current_case"] = None
        st.session_state["q_index"] = 0
        st.rerun()
    except Exception as e:
        st.warning(f"答题页面异常：{str(e)}")


def render_sidebar(player: PlayerRecord):
    """侧边栏：玩家信息、成就列表，原有展示逻辑不变，增加空值校验"""
    try:
        st.sidebar.header("👤 玩家信息")
        st.sidebar.write(f"玩家ID：{player.player_id}")
        st.sidebar.write(f"累计总分：{player.score}")
        st.sidebar.write(f"已完成病例：{len(player.finished_cases)} / {len(CASE_LIBRARY)}")

        st.sidebar.divider()
        st.sidebar.subheader("🏅 我的成就")
        if len(player.achievements) == 0:
            st.sidebar.caption("暂未解锁成就")
        else:
            for ach_key in player.achievements:
                info = ACHIEVEMENTS.get(ach_key, {})
                icon = info.get("icon", "")
                name = info.get("name", ach_key)
                desc = info.get("desc", "")
                st.sidebar.markdown(f"{icon} **{name}**：{desc}")

        st.sidebar.divider()
        if st.sidebar.button("🔄 重置当前玩家数据"):
            player.score = 0
            player.finished_cases = []
            player.achievements = []
            st.session_state["player_map"][player.player_id] = player
            st.sidebar.success("玩家数据已重置")
            time.sleep(1)
            st.rerun()
    except Exception as e:
        st.sidebar.warning(f"侧边栏渲染异常：{str(e)}")


def render_home_page():
    """首页：玩家登录、病例选择，原有选择逻辑不变，增加异常兜底"""
    try:
        st.title("🩺 医学病例模拟游戏")
        st.markdown("欢迎进入病例诊疗考核，选择病例开始答题，完成挑战解锁成就。")
        st.divider()

        # 玩家输入
        pid = st.text_input("请输入你的玩家ID（用于数据隔离）", value="")
        if not pid.strip():
            st.info("请输入玩家ID后进入游戏")
            return

        player = get_or_create_player(pid.strip())
        st.session_state["current_player"] = player

        render_sidebar(player)

        st.subheader("📁 病例列表")
        for case in CASE_LIBRARY:
            cid = case.get("case_id", "")
            title = case.get("title", "未知病例")
            done_flag = cid in player.finished_cases
            label_text = f"{title} {'✅已完成' if done_flag else ''}"

            if st.button(label_text, key=f"start_{cid}"):
                # 启动病例，初始化病例状态
                st.session_state["current_case"] = case
                st.session_state["q_index"] = 0
                st.session_state["case_score"] = 0
                st.session_state["case_start_ts"] = time.time()
                st.session_state["show_hint"] = False
                if case.get("ecg_enable", False):
                    st.session_state["ecg_running"] = True
                    st.session_state["ecg_phase"] = 0
                st.rerun()
    except Exception as e:
        st.warning(f"首页渲染异常：{str(e)}")


def main():
    """主函数入口"""
    try:
        init_session_state()
        current_case = st.session_state.get("current_case")
        if current_case is not None:
            player = st.session_state.get("current_player")
            if player is None:
                st.warning("玩家信息丢失，返回首页")
                st.session_state["current_case"] = None
                st.rerun()
                return
            render_sidebar(player)
            render_case_page(player)
        else:
            render_home_page()
    except Exception as e:
        st.error(f"程序主流程异常：{str(e)}")


if __name__ == "__main__":
    main()
