# RAG 文档智能问答系统

面向 **岗位 JD 与个人材料** 的检索增强生成（RAG）问答系统：把简历、项目介绍、岗位说明等文档放进知识库，就能用自然语言提问，系统会检索相关片段并调用大模型生成带引用来源的回答。

项目使用 FastAPI + Sentence-Transformers + FAISS 构建，大模型调用被抽象为标准的 Chat Completions HTTP 接口，可对接 OpenAI 及任意兼容该协议的服务（如 Moonshot Kimi、DeepSeek、通义千问等国产大模型网关）。

## 项目背景

在投递大模型 / RAG 相关岗位时，常常需要反复向面试官解释"我做过什么项目、简历上这句话具体指什么"。这个项目的初衷是把 **岗位 JD、个人简历、项目说明** 这些原本静态的文本，变成一个可以被"提问"的知识库：

- 面试官或 HR 可以直接问"这个人做过哪些相关项目？"
- 自己也可以用它验证"我的简历表述是否覆盖了 JD 的关键要求？"

同时，这也是一个完整的、可独立运行的 RAG 工程实践：从文档解析、切分、向量化、索引构建，到检索、Prompt 拼接、大模型生成、来源可追溯，覆盖了大模型应用工程化的核心链路。

## 核心功能

- 📄 **多格式文档接入**：支持 `.txt` / `.md` / `.pdf`，支持嵌套目录批量导入
- ✂️ **可配置文本切分**：按字符数切分并保留重叠窗口，避免语义在切分边界丢失
- 🧠 **中文语义向量化**：默认使用 `BAAI/bge-small-zh-v1.5`，专为中文检索优化
- 🔍 **FAISS 向量检索**：本地向量索引 + JSONL 元数据侧库，检索速度快、无外部依赖
- 💬 **RAG 问答与来源溯源**：回答基于检索到的上下文生成，并返回来源文件、chunk 编号与相关性分数，便于核实答案依据
- 🔌 **LLM 供应商无关**：通过标准 Chat Completions 协议对接任意兼容网关，不绑定单一模型厂商
- 🛡️ **配置与密钥分离**：所有参数通过环境变量注入，代码中不出现任何密钥或模型端点硬编码
- ⚠️ **健壮的异常处理**：文档缺失、索引未构建、LLM 调用失败等场景均返回明确的 HTTP 错误信息，而非静默失败或整体崩溃

## 技术架构

```
data/raw/                -> 原始文档目录（.txt / .md / .pdf）
app/document_loader.py   -> 读取原始文件，统一转换为纯文本
app/text_splitter.py     -> 按 chunk_size / chunk_overlap 切分文本
app/embedding.py         -> Sentence-Transformers 向量化（单例模型，避免重复加载）
app/vector_store.py      -> FAISS 索引 + JSONL 元数据侧库，负责持久化与检索
app/llm.py               -> Chat Completions 兼容协议的 HTTP 客户端
app/rag_pipeline.py      -> 编排 build_index / ask 的完整流程
app/main.py              -> FastAPI 路由层
app/config.py            -> 从环境变量集中读取全部配置
```

**数据流（问答链路）**：
用户问题 → Embedding 向量化 → FAISS Top-K 检索 → 拼接检索片段为上下文 → 组装 System/User Prompt → 调用大模型 → 返回 `answer` + `sources`

## 接口说明

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/build_index` | 扫描 `data/raw`，切分并向量化文档，(重新)构建 FAISS 索引 |
| `POST` | `/ask` | 提交问题，返回大模型答案及检索到的来源片段 |
| `GET`  | `/health` | 健康检查，返回索引是否已加载及索引规模 |

## 1. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 配置环境变量

复制示例文件并填入你自己的配置。**不要提交真实的 API Key，`.env` 已在 `.gitignore` 中忽略。**

```bash
cp .env.example .env
```

`.env` 示例（以下 Key 均为占位符，不是真实凭证）：

```
API_BASE_URL=https://api.openai.com/v1        # 或任意 OpenAI 协议兼容网关
API_KEY=your-api-key-here                      # 你的密钥，不要提交到版本库
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL_NAME=BAAI/bge-small-zh-v1.5
```

`API_BASE_URL` 需要指向一个实现了 `/chat/completions` 接口、且请求/响应结构与 OpenAI Chat Completions 协议兼容的服务端点。

## 3. 准备示例文档

将 `.txt` / `.md` / `.pdf` 文件放入 `data/raw/`（支持嵌套子目录）。项目自带三份示例文档，可直接用于体验完整流程：

```
data/raw/
├── rag_project.md         # 本项目的技术说明（架构、接口、工程化重点）
├── candidate_summary.md   # 示例候选人简历摘要（已脱敏）
└── job_description.md     # 示例岗位 JD（AIGC 工程实习生）
```

## 4. 启动服务

```bash
uvicorn app.main:app --reload --port 8000
```

## 5. 构建索引

```bash
curl -X POST http://localhost:8000/build_index | python -m json.tool
```

返回示例：

```json
{
    "status": "ok",
    "documents_indexed": 3,
    "chunks_indexed": 3
}
```

该接口会扫描 `data/raw`，将文档切分为若干 chunk，用配置的 Embedding 模型向量化，并把 FAISS 索引与元数据侧库写入 `vector_store/`。每次新增或修改文档后，需要重新调用该接口。

## 6. 提问

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "这个岗位需要哪些技能？", "top_k": 4}' \
  | python -m json.tool
```

返回示例：

```json
{
    "answer": "该岗位（AIGC 工程实习生）关注 LLM 应用、AIGC 生产线、知识库、RAG 和后端系统建设，要求具备 Python 或 Golang 编程能力，了解大模型应用开发、Prompt Engineering、数据处理与工程化开发经验 [job_description.md#0]。",
    "sources": [
        {
            "source": "job_description.md",
            "chunk_index": 0,
            "text": "该岗位关注 LLM 应用、AIGC 生产线、知识库、RAG、后端系统和任务流水线建设。候选人需要具备 Python 或 Golang 编程能力……",
            "score": 0.87
        }
    ]
}
```

再试一个跨文档的问题：

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "候选人的项目经历是否匹配这个岗位的要求？", "top_k": 4}' \
  | python -m json.tool
```

`top_k` 为可选参数，缺省时使用环境变量 `TOP_K`（默认 4）。

## 7. 健康检查

```bash
curl http://localhost:8000/health | python -m json.tool
```

```json
{
    "status": "ok",
    "index_loaded": true,
    "index_size": 3
}
```

## 配置说明

所有配置项均通过环境变量读取（完整列表见 `app/config.py` 与 `.env.example`），包括文本切分的 chunk 大小/重叠、检索 `top_k`、以及 LLM 请求的超时时间/温度/最大 token 数。源码中不存在任何硬编码的密钥或模型端点。

## 异常处理

- `data/raw` 中缺失或不支持的文件会被记录日志并跳过，不会导致整个索引构建失败
- `/build_index` 和 `/ask` 在失败时返回 HTTP 400，并在 `detail` 字段中给出具体原因（如未找到文档、索引尚未构建、大模型接口报错等）
- `app/llm.py` 对缺失 `API_BASE_URL`、请求超时、非 2xx 响应、响应结构异常等情况均抛出明确的错误信息

## 项目亮点

- **工程化优先**：配置与代码分离、密钥不落盘不硬编码、分层清晰（加载 → 切分 → 向量化 → 索引 → 检索 → 生成），每一层都可以独立替换或测试
- **面向中文场景优化**：Embedding 默认使用中文语义模型 `BAAI/bge-small-zh-v1.5`，比通用英文模型在中文检索任务上召回更准确
- **答案可溯源**：不是简单地把答案丢给用户，而是同时返回来源文件、chunk 编号和相关性分数，符合企业级 RAG 系统对"可解释性"的要求
- **LLM 供应商解耦**：基于标准 Chat Completions 协议封装，切换模型厂商（OpenAI / Moonshot / DeepSeek 等）无需改动业务代码
- **真实场景切入点小而完整**：用"岗位 JD + 个人材料"这个具体、可解释的场景，完整覆盖了 RAG 系统从 0 到 1 的关键环节

## 可扩展方向

- **检索增强**：引入混合检索（BM25 + 向量）、重排序（Rerank）模型，提升复杂问题的召回质量
- **多轮对话**：在 `/ask` 中加入历史对话上下文，支持追问和澄清
- **流式返回**：将 `/ask` 改造为 SSE / WebSocket 流式输出，提升前端交互体验
- **评测体系**：引入 RAGAS 等评测框架，量化衡量检索命中率、答案忠实度（Faithfulness）等指标
- **向量库替换**：将 FAISS 替换为 Milvus / Qdrant 等生产级向量数据库，支持增量更新与分布式部署
- **权限与多知识库**：支持多租户、多知识库隔离，按用户/角色控制可检索的文档范围
- **前端界面**：搭配一个简单的 Web 聊天界面，替代当前纯 API 的交互方式

## 适合写入简历的项目描述

> **RAG 文档智能问答系统**（Python / FastAPI / FAISS / Sentence-Transformers）
> 独立设计并实现了一套面向中文场景的检索增强生成（RAG）问答系统：支持多格式文档解析与切分、基于 `BAAI/bge-small-zh-v1.5` 的语义向量化、FAISS 向量检索，并通过标准 Chat Completions 协议对接大模型生成带来源引用的答案。系统采用环境变量驱动的配置管理，密钥与代码完全分离，具备完善的异常处理与健康检查机制，体现了对大模型应用工程化、Prompt Engineering 和后端接口设计的综合实践能力。
