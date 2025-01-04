import re
from typing import List, Dict
from langchain.output_parsers import PydanticOutputParser
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
import numpy as np
import faiss
from pydantic import BaseModel, Field

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


# FAISS 索引检索函数
def search_faiss_index(index, metadata, query_vector, top_k=5):
    """在 FAISS 索引中检索最相似的向量。"""
    query_vector = np.array(query_vector).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [
        {"metadata": metadata[idx], "distance": distances[0][i]}
        for i, idx in enumerate(indices[0])
    ]
    return results


# 构造现代文句子列表
def construct_modern_lines(poem: Poem) -> Poem:
    """根据诗句和现代翻译构建现代句子列表"""
    modern_sentences = [sentence.strip() for sentence in re.split(r'[；。]', poem.modern_translation) if sentence.strip()]
    modern_lines = []
    current_index = 0

    for line in poem.lines:
        punctuation_count = line.count("。") + line.count("；")
        matched_sentences = modern_sentences[current_index:current_index + punctuation_count]
        current_index += punctuation_count
        modern_line = "；".join(matched_sentences) + "。" if matched_sentences else ""
        modern_lines.append(modern_line)

    if current_index < len(modern_sentences):
        print(f"警告：现代文未完全分割，剩余句子：{modern_sentences[current_index:]}")
    elif current_index > len(modern_sentences):
        print(f"警告：现代文句子数量不足，部分古文未能匹配翻译。")

    poem.modern_lines = modern_lines
    return poem


# 翻译模块核心逻辑
def translate_poem(
    input_text: str,
    llm: LLM,
    index: faiss.Index,
    metadata: Dict,
    encoder: SentenceTransformer,
    templates: Dict[str, PromptTemplate]
) -> Poem:
    """
    执行翻译逻辑，根据 input_text 返回结构化的 Poem 对象。

    Args:
        input_text (str): 输入的古诗文文本。
        llm (LLM): 核心 LLM 模型。
        index (faiss.Index): FAISS 索引。
        metadata (Dict): 索引的元数据。
        encoder (SentenceTransformer): 编码器。
        templates (Dict[str, PromptTemplate]): 模板字典，包含结构解析、翻译和关键词提取模板。

    Returns:
        Poem: 包含翻译结果和解析结果的结构化 Poem 对象。
    """
    if not input_text.strip():
        raise ValueError("输入文本为空！")

    # 解构模板
    structure_template = templates["structure_template"]
    translate_template = templates["translate_template"]
    keyWordsParse_template = templates["keyWordsParse_template"]

    # 初始化输出解析器
    output_parser = PydanticOutputParser(pydantic_object=PoemStructure)

    # 解析诗歌结构
    print("正在解析诗歌结构...")
    structure_chain = structure_template | llm
    parsed_poem = output_parser.parse(structure_chain.invoke({"poem_text": input_text}))
    lines = [line.strip() for line in parsed_poem.lines if line.strip()]

    poem = Poem(
        title=parsed_poem.title,
        author=parsed_poem.author,
        lines=lines,
        modern_translation="",
        keywords_analysis={}
    )
    print("诗歌结构解析成功！")

    # 检索增强
    print("正在检索平行语料库...")
    translate_reference = ""
    for line in poem.lines:
        line_vector = encoder.encode(line)
        search_results = search_faiss_index(index, metadata, line_vector, top_k=4)
        translate_reference += "\n".join(
            [f"古文：{result['metadata']['classical_text']}；现代文：{result['metadata']['modern_text']}" for result in search_results]
        )

    # 翻译
    print("正在生成翻译结果...")
    translate_chain = translate_template | llm
    modern_translation = translate_chain.invoke({
        "context": translate_reference,
        "question": input_text
    }).strip()
    poem.modern_translation = modern_translation
    print("翻译生成成功！")

    # 构造现代文句子列表
    poem = construct_modern_lines(poem)

    # 提取关键词
    print("正在提取关键词...")
    for classical_sentence, modern_sentence in zip(poem.lines, poem.modern_lines):
        parallel_corpus = f"古文：{classical_sentence}\n现代文：{modern_sentence}"
        keyWordsParse_chain = keyWordsParse_template | llm
        keywords_result = keyWordsParse_chain.invoke({"line": parallel_corpus}).strip()

        keywords = {
            item.split(":")[0].strip(): item.split(":")[1].strip()
            for item in keywords_result.split(";")
            if ":" in item
        }
        poem.keywords_analysis.update(keywords)
    print("关键词提取成功！")

    return poem
