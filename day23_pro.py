import streamlit as st
from google import genai
from google.genai import types
import os

# --- 1. 页面装修 (Page Config) ---
# layout="wide" 是关键，开启宽屏模式，利用你截图里那么大的屏幕
st.set_page_config(
    page_title="Gemini Pro Max",
    page_icon="",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. 注入自定义 CSS (黑客技巧) ---
# Streamlit 允许你写一点点 CSS 来微调样式
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认的右上角菜单 */
    #MainMenu {visibility: hidden;}
    /* 隐藏底部的 'Made with Streamlit' */
    footer {visibility: hidden;}
    /* 调整一下聊天气泡的字体大小 */
    .stChatMessage {font-size: 1.1em;}
</style>
""", unsafe_allow_html=True)

st.title(" Gemini 全栈交互终端")
st.caption(" Powered by Google GenAI SDK v1.0 | Streamlit UI")

# --- 3. 侧边栏装修 (Control Panel) ---
with st.sidebar:
    st.title(" 控制面板")
    
    # API Key 输入 (使用 password 模式隐藏)
    # 如果系统环境变量里有，就自动填入，省得每次输入
    default_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.text_input("Gemini API Key", value=default_key, type="password")
    
    st.divider() # 画一条分割线，显得专业
    
    st.subheader("模型配置")
    
    # 模型选择器 (Dropdown)
    selected_model = st.selectbox(
        "选择模型",
        ["gemini-3-flash", "gemini-2.5-flash-lite", "gemini-2.5-flash"],
        index=0
    )
    
    # 参数滑块
    temperature = st.slider("随机性 (Temperature)", 0.0, 2.0, 0.7, help="值越高，AI 越疯癫")
    
    st.divider()
    
    st.subheader("大脑设定 (System Prompt)")
    # 文本区域，允许你实时修改 System Instruction
    system_instruction = st.text_area(
        "定义 AI 人设",
        value="你是一个乐于助人的全栈开发助手，回答要简洁专业。",
        height=100
    )
    
    st.divider()
    
    # --- 挑战任务答案：清空记忆按钮 ---
    # use_container_width=True 让按钮充满侧边栏，更好看
    if st.button(" 清空对话记录", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun() # 立即刷新页面

# --- 4. 逻辑核心 ---
if api_key:
    client = genai.Client(api_key=api_key)
else:
    st.warning(" 请在左侧输入 API Key 启动引擎")
    st.stop()

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. 渲染历史消息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. 处理输入 ---
if prompt := st.chat_input("输入指令..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 调用 AI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # 这里的 config 是新版 SDK 的写法
            response = client.models.generate_content(
                model=selected_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    system_instruction=system_instruction # 注入侧边栏的人设
                )
            )
            message_placeholder.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
        except Exception as e:
            st.error(f" 调用失败: {e}")
