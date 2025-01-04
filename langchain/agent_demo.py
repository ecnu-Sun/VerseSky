import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from typing import Optional, List,Dict
import requests
import os
from dotenv import load_dotenv
from module.translate import translate_poem  # 引入翻译模块
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
import faiss
import pickle
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class PoemStructure(BaseModel):
    title: str = Field(description="诗的题目")
    author: str = Field(description="诗的作者")
    lines: list[str] = Field(description="诗句的列表，每句作为一个字符串")

# 自定义 LLM 类
class SiliconFlowLLM(LLM):
    """自定义 LangChain LLM 类，适配 SiliconFlow 提供的 LLM 接口。"""

    api_key: str
    model: str
    api_base: str = "https://api.siliconflow.cn/v1"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        """返回 LLM 的类型标识。"""
        return "siliconflow"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 4096
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            raise ValueError(f"Error from API: {response.status_code}, {response.text}")

        return response.json()["choices"][0]["message"]["content"]

class CustomLLM(SiliconFlowLLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if "翻译" in prompt:
            prompt = (
                "你是一个文言文学习助手，你的任务是帮助用户翻译古文,赏析古文，回答用户的问题。"
                "注意：当用户提到翻译文言文时，必须调用 `Translate` 工具。\n"
                + prompt
            )
        return super()._call(prompt, stop)


# 加载 FAISS 索引和元数据
def load_faiss_index(index_path, metadata_path):
    """加载持久化的 FAISS 索引和元数据。"""
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# 初始化Streamlit应用
st.title("古诗词学习助手")
st.write("这个助手可以帮助你翻译和赏析古诗词，并支持简单的多轮对话！")

# 初始化状态
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.agent = None  # Agent

if not st.session_state.initialized:
    # 加载环境变量
    load_dotenv()

    # 初始化语言模型
    llm = CustomLLM(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        model="THUDM/glm-4-9b-chat"
    )

    # 初始化对话记忆
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # 初始化 FAISS 索引和句子编码器
    index_path = "translation_refer/faiss_index.bin"
    metadata_path = "translation_refer/metadata.pkl"
    index, metadata = load_faiss_index(index_path, metadata_path)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # 初始化输出解析器
    output_parser = PydanticOutputParser(pydantic_object=PoemStructure)

    # 初始化 PromptTemplates
    structure_template = PromptTemplate(
        input_variables=["poem_text"],
        template=(
            "将以下诗分解为包含题目、作者和句子列表的 JSON 对象，如果输入不包含题目和作者，请留空：\n\n"
            "请确保输出符合以下格式：\n\n"
            "{format_instructions}\n\n"
            "诗：\n"
            "{poem_text}"
        ),
        partial_variables={"format_instructions": output_parser.get_format_instructions()},  # 动态注入格式说明
    )
    translate_template = PromptTemplate(
        input_variables=["context", "question"],
        template="根据以下关键词参考,将古文翻译成现代汉语：\n\n参考：{context}\n\n古文：{question}\n\n翻译："
    )
    keyWordsParse_template = PromptTemplate(
        input_variables=["line"],
        template = (
            "以下是古文与现代文平行语料，请提取关键词对，格式要求如下：\n\n"
            "1. 用':'分隔古文关键词和现代文对应词。\n"
            "2. 每个关键词对以';'结尾。\n"
            "3. 仅提取核心词汇，不分解整句，不提取古今含义明显相同的词语\n\n"
            "4.若没有满足条件的关键词，可以不做提取"
            "5. 多句平行语料的提取结果之间无需分隔符\n\n"
            "6.如果关键词和现代文对应词完全相同，不要做提取！"
            "多句平行语料：\n{line}\n\n"
            "请直接按照上述格式输出提取结果。不做分点或额外标识"
        )
    )
    templates = {
        "structure_template": structure_template,
        "translate_template": translate_template,
        "keyWordsParse_template": keyWordsParse_template,
    }

    # 定义翻译工具
    def translation_tool(input_text: str) -> str:
        try:
            poem = translate_poem(input_text, llm, index, metadata, encoder, templates)
            return {"status": "success", "translation": poem.modern_translation}

        except Exception as e:
            return f"翻译失败：{e}"
    
    # 创建工具列表
    tools = [
        Tool(
        name="Translate",
        func=translation_tool,
        description="翻译古诗文并返回结构化数据，包括标题、作者、原文、现代翻译、逐句翻译和关键词解析。"
    )
    ]

    # 初始化Agent
    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="conversational-react-description",
        memory=memory,
        verbose=True
    )

    st.session_state.initialized = True


# 用户输入
user_input = st.text_area("请输入古诗文或你的问题：", placeholder="例如：翻译‘明月松间照，清泉石上流’")

if st.button("提交"):
    if user_input.strip():
        with st.spinner("处理中，请稍候..."):
            try:
                # 调用Agent处理用户输入
                response = st.session_state.agent.invoke({"input": user_input})

                # 显示结果
                st.success("任务完成！")
                st.write("### 助手回复：")
                st.write(response)

            except Exception as e:
                st.error(f"执行失败：{e}")
    else:
        st.warning("请输入有效的文本！")
