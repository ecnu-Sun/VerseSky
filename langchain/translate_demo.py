from langchain.llms.base import LLM
from typing import Optional, List
import requests
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
import streamlit as st
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
import re
import torchvision
# 禁用 Beta 功能相关的警告

torchvision.disable_beta_transforms_warning()

class PoemStructure(BaseModel):
    title: str = Field(description="诗的题目")
    author: str = Field(description="诗的作者")
    lines: list[str] = Field(description="诗句的列表，每句作为一个字符串")

# 定义目标 JSON 的数据结构
class Poem(BaseModel):
    title: str = Field(description="诗的题目")
    author: str = Field(description="诗的作者")
    lines: list[str] = Field(description="诗句的列表，每句作为一个字符串")
    modern_translation: str = Field(default="", description="现代文翻译")
    modern_lines: list[str] = Field(default=[], description="现代文句子列表，每句作为一个字符串")
    keywords_analysis: dict = Field(default_factory=dict, description="关键词解析，存储古文关键词和现代文翻译的键值对")


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


def load_faiss_index(index_path, metadata_path):
    """加载持久化的 FAISS 索引和元数据。"""
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search_faiss_index(index, metadata, query_vector, top_k=5):
    """在 FAISS 索引中检索最相似的向量。"""
    query_vector = np.array(query_vector).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [
        {"metadata": metadata[idx], "distance": distances[0][i]}
        for i, idx in enumerate(indices[0])
    ]
    return results

def construct_modern_lines(poem):
    """
    根据 poem.lines 和 poem.modern_translation，构建 poem.modern_lines。
    假设 modern_translation 中的句号与 lines 中的句号数量对应，并考虑中文分号"；"的断句作用。

    Args:
        poem: 包含 lines 和 modern_translation 属性的对象。

    Returns:
        poem: 更新后的 poem 对象，增加 modern_lines 属性。
    """
    # 将 modern_translation 按句号和分号拆分为句子列表
    modern_sentences = [sentence.strip() for sentence in re.split(r'[；。]', poem.modern_translation) if sentence.strip()]

    # 初始化结果
    modern_lines = []
    current_index = 0  # 用于跟踪 modern_sentences 的分割进度

    # 遍历 poem.lines，根据句号数量划分 modern_sentences
    for line in poem.lines:
        # 统计当前古文 line 的句号数量
        punctuation_count = line.count("。") + line.count("；")

        # 根据标点数量截取对应的 modern_sentences
        matched_sentences = modern_sentences[current_index:current_index + punctuation_count]
        current_index += punctuation_count  # 更新索引位置

        # 将匹配的现代文句子组合为一段
        modern_line = "；".join(matched_sentences) + "。" if matched_sentences else ""
        modern_lines.append(modern_line)

    # 如果分割过程中发现句号数量不匹配，记录警告
    if current_index < len(modern_sentences):
        print(f"警告：现代文未完全分割，剩余句子：{modern_sentences[current_index:]}")
    elif current_index > len(modern_sentences):
        print(f"警告：现代文句子数量不足，部分古文未能匹配翻译。")

    # 更新 poem 对象
    poem.modern_lines = modern_lines
    return poem




load_dotenv()  # 加载环境变量

# Streamlit 应用逻辑
st.title("文言文翻译助手")
st.subheader("使用 SiliconFlow LLM 翻译文言文")

# 初始化阶段
if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False
    st.session_state.index = None
    st.session_state.metadata = None
    st.session_state.encoder = None

# 仅初始化一次
if not st.session_state.is_initialized:
    st.write("### 系统初始化中，请稍候...")
    progress_bar = st.progress(0)

    try:
        # Step 1: 加载 FAISS 索引
        st.info("正在加载 FAISS 索引...")
        index_path = "translation_refer/faiss_index.bin"
        metadata_path = "translation_refer/metadata.pkl"
        st.session_state.index, st.session_state.metadata = load_faiss_index(index_path, metadata_path)
        progress_bar.progress(50)
        st.success("FAISS 索引加载成功！")
        # Step 2: 加载编码模型
        st.info("正在加载编码模型...")
        st.session_state.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        progress_bar.progress(100)
        st.success("系统初始化完成！")
        st.session_state.is_initialized = True

    except Exception as e:
        st.error(f"初始化失败：{e}")
        st.stop()


# 翻译逻辑
input_text = st.text_area("请输入文言文", placeholder="例如：明月松间照，清泉石上流。")

if st.button("开始翻译"):
    if not input_text.strip():
        st.error("请输入文言文！")
    else:
        with st.spinner("正在执行翻译任务..."):
            try:
                output_parser = PydanticOutputParser(pydantic_object=PoemStructure)
                progress_bar = st.progress(0)
                # 配置工作链
                core_llm = SiliconFlowLLM(
                    api_key=os.getenv("SILICONFLOW_API_KEY"),
                    model="THUDM/glm-4-9b-chat"
                )
                assist_llm = SiliconFlowLLM(
                    api_key=os.getenv("SILICONFLOW_API_KEY"),
                    model="THUDM/chatglm3-6b"
                )
                translate_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="根据以下关键词参考,将古文翻译成现代汉语：\n\n参考：{context}\n\n古文：{question}\n\n翻译："
                )

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

                st.info("正在解析诗歌结构...")
                # 结构化解析诗歌
                poemSturcture_chain = structure_template | core_llm
                parsed_poem = output_parser.parse(poemSturcture_chain.invoke({"poem_text": input_text}))
                # 过滤掉空的 line
                parsed_poem.lines = [line.strip() for line in parsed_poem.lines if line.strip()]

                # 构造完整poem对象
                poem = Poem(
                    title=parsed_poem.title,
                    author=parsed_poem.author,
                    lines=parsed_poem.lines,
                    modern_translation="",
                    keywords_analysis={}
                )
                progress_bar.progress(25)
                st.success("诗歌结构解析成功！")

                st.info("正在检索平行语料库...")
                # 检索增强
                translate_reference = ""
                # 为每一句诗检索相似的平行语料
                for line in poem.lines:
                    st.write("### 当前句子: "+line)
                    line_vector = st.session_state.encoder.encode(line)
                    search_results = search_faiss_index(
                        st.session_state.index,
                        st.session_state.metadata,
                        line_vector,
                        top_k=4
                    )
                    
                    # 整合成 古文-现代文 平行语料
                    relative_poemLine = "\n".join(
                        [f"古文：{result['metadata']['classical_text']}\t 现代文：{result['metadata']['modern_text']};" for result in search_results]
                    )
                    st.write(relative_poemLine)
                    
                    # 将平行语料拆解为关键词
                    keyWordsParse_chain = keyWordsParse_template | core_llm
                    translate_refer = keyWordsParse_chain.invoke({"line": relative_poemLine})
                    # 过滤掉 key=value 的重复关键词对
                    translate_refer = ";".join(
                        pair for pair in translate_refer.split(";")
                        if ":" in pair and pair.split(":")[0].strip() != pair.split(":")[1].strip()
                    )

                    st.write("\n"+translate_refer)
                    # 整合所有句子的关键词解释
                    translate_reference += translate_refer
                st.success("平行语料库检索成功！")
                st.write(translate_reference)
                progress_bar.progress(50)
                # 翻译链
                translation_chain = translate_template | core_llm
                poem_text = "\n".join(poem.lines)
                # 利用参考关键词执行翻译任务
                st.info("正在生成翻译结果...")

                result = translation_chain.invoke({"context": translate_reference, "question": poem_text})
                poem.modern_translation = result.strip()  # 保存翻译结果到对象
                st.success("翻译生成成功！")
                # 显示翻译结果
                st.write("### 翻译结果：")
                st.write(str(result))
                progress_bar.progress(75)

                # 整句拆分并生成逐句平行语料
                st.info("正在生成逐句平行语料并提取关键词...")

                # 生成现代文句子列表
                poem = construct_modern_lines(poem)

                # 初始化关键词存储
                keywords_analysis = {}

                # 逐句生成平行语料并提取关键词
                for i, (classical_sentence, modern_sentence) in enumerate(zip(poem.lines, poem.modern_lines)):
                    st.write(f"### 第{i + 1}句平行语料：")
                    parallel_corpus = f"古文：{classical_sentence}\n现代文：{modern_sentence}"
                    st.write(parallel_corpus)

                    # 调用 LLM 提取关键词
                    keyWordsParse_chain = keyWordsParse_template | core_llm
                    try:
                        keywords_result = keyWordsParse_chain.invoke({"line": parallel_corpus})
                        # 过滤掉 key=value 的重复关键词对
                        keywords_result = ";".join(
                            pair for pair in keywords_result.split(";")
                            if ":" in pair and pair.split(":")[0].strip() != pair.split(":")[1].strip()
                        )
                        st.write(f"关键词提取结果：{keywords_result}")

                        # 解析关键词并保存
                        keywords = {
                            item.split(":")[0].strip(): item.split(":")[1].strip()
                            for item in keywords_result.split(";") if ":" in item
                        }
                        for key, value in keywords.items():
                            if key not in keywords_analysis:  # 避免覆盖已有关键词
                                keywords_analysis[key] = value
                    except Exception as e:
                        st.error(f"关键词提取失败（第{i + 1}句）：{e}")

                # 保存关键词解析到 Poem 对象
                poem.keywords_analysis = keywords_analysis
                st.success("关键词提取成功！")
                st.write("### 关键词解析结果：")
                st.json(poem.keywords_analysis)
                st.json(poem.model_dump_json())
                progress_bar.progress(100)


            except ValueError as e:
                st.error(f"翻译失败：{e}")
            except requests.exceptions.RequestException as e:
                st.error(f"网络请求错误：{e}")
