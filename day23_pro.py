import streamlit as st
from google import genai
from google.genai import types
import os
from typing import TypedDict
from langgraph.graph import StateGraph, END

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="Gemini 智能体工厂",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Gemini 深度思考 Agent")
st.caption("集成 LangGraph：写作 -> 反思 -> 修正 自动化闭环")

# --- 2. 侧边栏配置 ---
with st.sidebar:
    st.header(" 控制台")
    # API Key 管理
    default_key = os.environ.get("GEMINI_API_KEY", "")
    # 如果 secrets 里有，优先用 secrets
    if "GEMINI_API_KEY" in st.secrets:
        default_key = st.secrets["GEMINI_API_KEY"]
        
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    
    st.divider()
    model_name = st.selectbox("选择模型", ["gemini-2.5-flash", "gemini-2.5-flash-lite"], index=0)
    max_revisions = st.slider("最大反思次数", 1, 5, 2, help="批评家最多可以让作家重写几次？")
    
    with st.expander(" 角色设定 (高级)"):
        writer_instruction = st.text_area("作家设定", value="你是一个严谨的技术作家，善于使用简单的语言解释复杂的概念。")
        critic_instruction = st.text_area("批评家设定", value="你是一个吹毛求疵的审核员，不仅检查事实错误，还关注逻辑连贯性和语气。")
    
    if st.button(" 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 3. 初始化 Client ---
if not api_key:
    st.warning("请先配置 API Key")
    st.stop()

client = genai.Client(api_key=api_key)

# --- 4. 定义 LangGraph 逻辑 (Day 24 的核心) ---

# 定义状态
class AgentState(TypedDict):
    task: str
    draft: str
    critique: str
    revision_count: int
    content_history: list # 用来记录中间过程，方便在网页展示
    writer_instruction: str
    critic_instruction: str

# 定义节点 A：作家
def writer_node(state: AgentState):
    task = state['task']
    critique = state.get('critique', "")
    count = state.get('revision_count', 0)
    history = state.get('content_history', [])
    writer_instruction = state.get('writer_instruction', "")
    
    if count == 0:
        prompt = f"""
        【你的角色设定】：{writer_instruction}
        【任务】：请简短地写一段关于 '{task}' 的介绍。
        """
        step_name = " 初稿创作中..."
    else:
        prompt = f"""
        【你的角色设定】：{writer_instruction}
        原稿：{state['draft']}
        批评意见：{critique}
        任务：请根据批评意见，重写这段关于 '{task}' 的介绍。
        """
        step_name = f" 第 {count+1} 次修改中..."
        
    response = client.models.generate_content(
        model=model_name, contents=prompt
    )
    
    # 记录过程
    history.append(f"**{step_name}**\n\n{response.text}")
    
    return {
        "draft": response.text, 
        "revision_count": count + 1,
        "content_history": history
    }

# 定义节点 B：批评家
def critic_node(state: AgentState):
    draft = state['draft']
    history = state.get('content_history', [])
    critic_instruction = state.get('critic_instruction', "")
    
    prompt = f"""
    【你的角色设定】：{critic_instruction}
    
    请审核以下草稿：
    {draft}
    
    如果草稿写得非常完美且字数超过 50 字，请回复 'PASS'。
    如果草稿太短或者有错误，请给出简短的修改建议（不要超过 20 字）。
    """
    
    response = client.models.generate_content(
        model=model_name, contents=prompt
    )
    
    history.append(f"**🧐 批评家审核:** {response.text}")
    
    return {
        "critique": response.text,
        "content_history": history
    }

# 定义路由逻辑
def should_continue(state: AgentState):
    critique = state['critique']
    count = state['revision_count']
    
    # 这里用侧边栏的 max_revisions 变量
    if "PASS" in critique or count >= max_revisions:
        return END
    return "writer"

# 构建图 (放到函数里，每次调用时构建)
def get_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.set_entry_point("writer")
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges("critic", should_continue, {END: END, "writer": "writer"})
    return workflow.compile()

# --- 5. 聊天界面逻辑 ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果有中间思考过程，用折叠面板显示
        if "thoughts" in msg:
            with st.expander("查看 AI 的思考/反思过程"):
                for step in msg["thoughts"]:
                    st.markdown(step)
                    st.divider()

# 处理输入
if prompt := st.chat_input("输入一个主题（例如：Python语言、量子力学...）"):
    # 1. 显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 运行 LangGraph
    with st.chat_message("assistant"):
        status_container = st.status("🧠 AI 正在进行深度思考循环...", expanded=True)
        
        try:
            app = get_graph()
            inputs = {
                "task": prompt, 
                "revision_count": 0, 
                "content_history": [],
                "writer_instruction": writer_instruction,
                "critic_instruction": critic_instruction
            }
            
            # 运行图，拿到最终状态
            final_state = app.invoke(inputs)
            
            # 更新状态容器
            status_container.update(label=" 思考完成！", state="complete", expanded=False)
            
            # 显示最终结果
            final_response = final_state['draft']
            st.markdown(final_response)
            
            # 拿到中间过程历史
            thoughts = final_state['content_history']
            
            # 在折叠面板里展示中间过程（让用户看到Writer和Critic的吵架过程）
            with st.expander("点击查看 作家 vs 批评家 的博弈过程"):
                for step in thoughts:
                    st.markdown(step)
                    st.divider()

            # 保存到历史
            st.session_state.messages.append({
                "role": "assistant", 
                "content": final_response,
                "thoughts": thoughts # 把思考过程也存下来
            })
            
        except Exception as e:
            status_container.update(label=" 出错了", state="error")
            st.error(f"运行失败: {e}")
