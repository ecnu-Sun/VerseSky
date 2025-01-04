import os
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import pickle


def build_and_save_faiss_index(file_path, index_path, batch_size=512, use_gpu=True):
    """
    加载平行文本并构建持久化 FAISS 向量索引，支持 GPU 加速和进度条。
    :param file_path: 平行文本文件路径，每行为“现代文\t古文”。
    :param index_path: 持久化 FAISS 索引的保存路径。
    :param batch_size: 每批处理的文本数量。
    :param use_gpu: 是否启用 GPU 加速 FAISS。
    """
    # 加载平行文本数据
    documents = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            if "\t" in line:  # 现代文和古文之间用制表符分隔
                modern_text, classical_text = line.strip().split("\t")
                documents.append({
                    "modern_text": modern_text,  # 现代文
                    "classical_text": classical_text  # 古文
                })
    print(f"成功加载 {len(documents)} 条平行文本数据！")

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在使用设备：{device}")

    # 加载嵌入模型
    print("正在加载嵌入模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )
    print("嵌入模型加载完成！")

    # 批量生成嵌入并显示进度条
    print("正在生成嵌入向量...")
    classical_texts = [doc["classical_text"] for doc in documents]  # 提取古文
    embeddings = []  # 存储古文的嵌入向量
    metadata = []  # 存储“现代文-古文”对

    for i in tqdm(range(0, len(classical_texts), batch_size), desc="生成嵌入进度"):
        batch_texts = classical_texts[i:i+batch_size]  # 当前批次古文
        batch_embeddings = embedding_model.embed_documents(batch_texts)  # 生成嵌入向量
        embeddings.extend(batch_embeddings)  # 存储嵌入
        metadata.extend(documents[i:i+batch_size])  # 对应存储“现代文-古文”元数据

    print("嵌入向量生成完成！")

    # 构建 FAISS 向量索引
    print("正在构建 FAISS 索引...")
    embedding_dim = len(embeddings[0])  # 嵌入向量的维度
    index = faiss.IndexFlatL2(embedding_dim)  # 使用 L2 距离的平面索引

    if use_gpu and torch.cuda.is_available():
        print("将索引迁移到 GPU 加速...")
        try:
            res = faiss.StandardGpuResources()  # GPU 资源
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except AttributeError:
            print("错误：您的 FAISS 安装不支持 GPU，请安装 faiss-gpu 版本。")
            print("将退回到 CPU 模式...")
            use_gpu = False  # 回退到 CPU 模式

    # 添加数据到索引
    for i in tqdm(range(0, len(embeddings), batch_size), desc="构建索引进度"):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_embeddings = np.array(batch_embeddings)  # 转换为 NumPy 数组
        index.add(batch_embeddings)

    print("FAISS 索引构建完成！")

    # 保存索引
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "faiss_index.bin"))
    print(f"FAISS 索引已保存至 {index_path}/faiss_index.bin！")

    # 保存元数据
    metadata_path = os.path.join(index_path, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"元数据已保存至 {metadata_path}！")


if __name__ == "__main__":
    # 输入文件路径和输出索引路径
    input_file = "modern_to_classical.txt"  # 平行文本文件路径
    output_index = "translation_refer"  # 保存索引的文件夹名称

    # 构建并保存索引
    build_and_save_faiss_index(input_file, output_index, batch_size=512, use_gpu=True)
