# 基于 RAG 的文档智能问答系统项目说明

本项目基于 Python、FastAPI、FAISS、Sentence Transformers 和大模型 API 构建，用于实现文档知识库问答。系统支持读取 Markdown、TXT、PDF 文件，并完成文档解析、文本切分、Embedding 向量化、FAISS 向量索引构建、Top-K 语义检索、Prompt 拼接和大模型答案生成。

系统提供三个主要接口：/health 用于健康检查，/build_index 用于构建知识库索引，/ask 用于接收用户问题并返回答案。/ask 接口会返回 answer 和 sources，其中 sources 包括来源文件、chunk 编号、相关性分数和原始文本片段，便于追踪答案依据。

项目重点关注大模型应用工程化中的知识库构建、RAG 检索增强生成、Prompt Engineering、后端接口封装、环境变量配置和异常处理，避免将 API Key 写入代码，提高系统可维护性和安全性。
