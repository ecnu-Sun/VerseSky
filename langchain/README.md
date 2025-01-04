# langchain 实现核心功能

## 结构
module/ : 存放langchain工具调用模块，目前实现了翻译模块translate.py

agent_app.py : agent模块，启动服务后，还负责与java后端交互。

agent_demo.py : agent开发测试， 启动指令：streamlit run agent_demo.py ，启动后模型加载需要1~2分钟。

translate_demo.py ：翻译模块开发测试， 启动指令：streamlit run translate_demo.py ，启动后模型加载需要1~2分钟。

parallel_texts_extract.py ：从“双语数据”中提取古今平行语料到 modern_to_classical.txt

build_faiss.py : 利用modern_to_classical.txt构建faiss(向量数据库)的索引和元数据，存储至 translation_refer/

## 依赖安装
创建虚拟环境：
python -m venv venv

激活虚拟环境：
venv\Scripts\activate

安装依赖：
pip install -r requirements.txt

依赖有5个G，主要是torch库较大。

## 核心逻辑
使用 chatGLM4 为agent的LLM，为用户提供多轮对话。
在agent_app.py中，agent会在process_input（）函数中接收java后端的访问，接收用户对话的输入。然后agent执行代理任务。

执行过程中可能会调用“翻译模块”。

**翻译模块会向java后端发送格式为：**
```python
{
    "action": "update_view", 
    "data": poem_data   # poem_data是诗歌翻译后的结构化数据
}
```
action字段向后端指明前端展示页面需要执行的行为。这里update_view指明前端需要展示翻译窗口，并在窗口中展示古今对比。

poem_data的字段定义如下：
```python
{
    "title": "诗的题目",
    "author": "诗的作者",
    "lines": [
        "诗句1",
        "诗句2",
        "诗句3"
    ],
    "modern_translation": "现代文翻译",
    "modern_lines": [
        "现代文句子1",
        "现代文句子2",
        "现代文句子3"
    ],
    "keywords_analysis": {
        "古文关键词1": "现代文翻译1",
        "古文关键词2": "现代文翻译2"
    }
}
```
poem_data 里的数据会被用在翻译窗口（古今对照学习）展示。除此之外，还可以为每个用户保存它学习（翻译）过的poem_data，以维护用户的学习过程。

**agent 的常规（多轮对话）回复格式是：**
```python
{
    "action": ""default_action"", 
    "data": agent_response   # response是直接由agent生成的回复
}
```

agent_response的字段定义如下：
```python
{
  "input": "string",
  "chat_history": [
    "string"
  ],
  "output": "string"
}

```

***一段示例如下：***

```python
{
  "input": "能帮我翻译下这篇古文吗：\n六王毕，四海一...",
  "chat_history": [
    "HumanMessage(content='你好', additional_kwargs={}, response_metadata={})",
    "AIMessage(content='你好，请问有什么可以帮助您的吗？', additional_kwargs={}, response_metadata={})",
    "HumanMessage(content='你会做什么？', additional_kwargs={}, response_metadata={})",
    "AIMessage(content='我可以回答各种问题，提供深入的解释和讨论...', additional_kwargs={}, response_metadata={})",
    "HumanMessage(content='能帮我翻译下这篇古文吗：六王毕，四海一...')",
    "AIMessage(content='这篇文言文是《阿房宫赋》，作者为唐代文学家杜牧...')"
  ],
  "output": "这篇文言文是《阿房宫赋》，作者为唐代文学家杜牧..."
}

```
这里chat_history可以用来维护用户的对话历史记录

## java后端任务：
需要利用agent向后端发送的 poem_data 和 agent_response,分别实现翻译内容的展示、多轮对话的维护（在数据库中保存用户对话等）、用户学习（翻译）过的诗词的记录。

