import uvicorn  # 用于运行 FastAPI 应用
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM
from module.translate import translate_poem  # 引入翻译模块
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from typing import Optional, List,Dict
import faiss
import pickle
import os
from dotenv import load_dotenv
import requests
from langchain.output_parsers import PydanticOutputParser
from langchain.agents import AgentExecutor
app = FastAPI()


# 定义请求数据模型
class UserInput(BaseModel):
    input_text: str

# 定义响应数据模型
class ResponseData(BaseModel):
    action: str
    data: dict  # 如果 `data` 的结构固定，可以进一步细化


# 定义诗歌结构数据模型
class PoemStructure(BaseModel):
    title: str
    author: str
    lines: list[str]

# 自定义 LLM 类，用于调用 SiliconFlow API
class SiliconFlowLLM(LLM):
    api_key: str
    model: str
    api_base: str = "https://api.siliconflow.cn/v1"
    temperature: float = 0.7

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def _call(self, prompt: str, stop=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 4096,
        }
        if stop:
            payload["stop"] = stop

        response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
        if response.status_code != 200:
            raise ValueError(f"Error from API: {response.status_code}, {response.text}")

        return response.json()["choices"][0]["message"]["content"]

# 定义agent所使用的LLM
class CustomLLM(SiliconFlowLLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        prompt = (
            "你是一个文言文学习助手，你的任务是帮助用户翻译古文,赏析古文，回答用户的问题。"
            "注意：当用户提到'翻译'文言文时，禁止直接给出翻译，'必须'调用 `Translate` 工具进行翻译。\n"
            + prompt
        )
        return super()._call(prompt, stop)


# 加载 FAISS 索引和元数据
def load_faiss_index(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# 加载 Faiss 索引和元数据
index_path = "translation_refer/faiss_index.bin"
metadata_path = "translation_refer/metadata.pkl"

# 初始化资源
load_dotenv()

print("开始加载 FAISS 索引...")
index, metadata = load_faiss_index(index_path, metadata_path)
print("FAISS 索引加载完成。")

print("开始加载 SentenceTransformer 模型...")
# 首次运行时，需要下载 SentenceTransformer 模型，之后可以直接加载
# encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# encoder.save("saved_model")

# 加载已保存的模型
encoder = SentenceTransformer("saved_model")
print("模型加载完成。")




# agent所使用的LLM
llm = CustomLLM(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    model="THUDM/glm-4-9b-chat"
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)




# 初始化 PromptTemplates
output_parser = PydanticOutputParser(pydantic_object=PoemStructure)
structure_template = PromptTemplate(
    input_variables=["poem_text"],
    template=(
        "将以下诗分解为包含题目、作者和句子列表的 JSON 对象，如果输入不包含题目和作者，请留空：\n\n"
        "请确保输出符合以下格式：\n\n"
        "{format_instructions}\n\n"
        "诗：\n"
        "{poem_text}"
    ),
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

translate_template = PromptTemplate(
    input_variables=["context", "question"],
    template="根据以下关键词参考,将古文翻译成现代汉语：\n\n参考：{context}\n\n古文：{question}\n\n翻译：",
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


# 翻译工具
def translation_tool(input_text: str) -> dict:
    try:
        # 调用翻译函数
        poem = translate_poem(input_text, llm, index, metadata, encoder, templates)
        
        # 准备返回的数据
        # poem_data = {
        #     "title": poem.title,
        #     "author": poem.author,
        #     "lines": poem.lines,
        #     "modern_translation": poem.modern_translation,
        #     "modern_lines": poem.modern_lines,
        #     "keywords_analysis": poem.keywords_analysis,
        # }

        # # 准备发送到 Java 后端的数据
        # java_backend_url = "http://your-java-backend/api/endpoint"      # 将 URL 替换为实际的 Java 后端接口地址
        # response = requests.post(java_backend_url, json={"action": "update_view", "data": poem_data})

        # # 发送失败，在对话中返回翻译信息
        # if response.status_code != 200:
        # return f"翻译成功，但发送数据到后端失败：{response.text}，以下是翻译结果：{poem_data.mordern_translation}"
        # return f"{poem.modern_translation}\n\n已经通过翻译工具得翻译结果，现在需要将翻译结果展示给用户"
        return {"status": "success", "translation": poem.modern_translation}

        # 发送成功，直接在对话中返回翻译成功消息即可
        # return "成功翻译，请查看翻译展示窗口"

    except Exception as e:
        return {"action": "error", "data": {"message": str(e)}}


tools = [
    Tool(
        name="Translate",
        func=translation_tool,
        description="翻译文言文，将得到的结果全部展示给用户，不能做任何分析。",
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="conversational-react-description",
    memory=memory,
    verbose=True,
    return_only_outputs=True,
    handle_parsing_errors=True,  # 捕获解析错误
)

# 定义 API 路由
@app.post("/process", response_model=ResponseData)
async def process_input(user_input: UserInput):
    input_text = user_input.input_text.strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="输入不能为空")
    try:
        response = agent.invoke({"input": input_text})

        # 如果返回已经满足格式，直接返回
        if isinstance(response, dict) and "action" in response:
            return response

        # 否则封装为默认返回格式
        return {
            "action": "default_action",  # 多轮对话的默认动作
            "data": response  
        }
    except Exception as e:
        return {"action": "error", "data": {"message": str(e)}}


if __name__ == "__main__":
    # 启动 FastAPI 服务
    uvicorn.run(
        "agent_app:app",  # 将 "your_script_name" 替换为实际的脚本文件名（不包含 .py 后缀）
        host="0.0.0.0",  # 绑定到所有网络接口，允许外部访问
        port=8000,       # 服务运行端口
        reload=True      # 自动重载（开发环境下启用）
    )