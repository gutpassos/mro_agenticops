# mro_agenticops

**AgenticOps | LLMOps + RAG Local | Offline-First Architecture**

Sistema de RAG (Retrieval-Augmented Generation) para documentos MRO (Modelo de Responsabilidade Organizacional) com execução 100% local usando Ollama + FAISS.

---

## 🏗️ Arquitetura

### Camadas do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    CAMADA AGENTIC (Futuro)                  │
│  Agentes autônomos | Planejamento | Ferramentas | Memória   │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  CAMADA DE APLICAÇÃO                        │
│  src/query.py    │ src/agent.py  │  prompts/                │
│  Orquestração RAG │ Lógica agentic │ Templates              │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 CAMADA DE RETRIEVAL                         │
│  src/retriever.py  │  src/embeddings.py │ src/rerank.py    │
│  FAISS index       │  Sentence-BERT     │ Reranking        │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CAMADA DE GERAÇÃO                         │
│      src/llm.py    │    Ollama API     │   Context Manager │
│      Llama 3.x     │    Local Server   │   Prompt Builder  │
└─────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  CAMADA DE DADOS                            │
│  src/preprocess.py │ src/chunking.py │ data/raw/*.pdf      │
│  PDF extraction    │ Semantic chunks │ Metadata enrichment │
└─────────────────────────────────────────────────────────────┘
```

### Componentes Principais

#### 1. **Data Pipeline** (`src/preprocess.py`, `src/train.py`)
- **Ingestão**: PyPDF2/pypdfium2 para extração de PDFs
- **Chunking**: Semantic chunking com sobreposição
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Indexação**: FAISS com índice IVF ou HNSW para escala

#### 2. **Retrieval Engine** (`src/retriever.py`)
- **Vector Store**: FAISS (offline, rápido)
- **Estratégia**: Híbrida (semantic + keyword BM25)
- **Reranking**: Cross-encoder para top-k refinement

#### 3. **LLM Integration** (`src/llm.py`)
- **Ollama API**: Cliente Python para modelos locais
- **Modelos suportados**: llama3, mistral, mixtral
- **Streaming**: Respostas progressivas para UX

#### 4. **Observability** (`monitoring/`)
- **MLflow**: Tracking de experimentos e métricas RAG
- **Logs estruturados**: JSON para análise
- **Métricas**: Latência, relevância, custo de contexto

---

## 📁 Estrutura de Diretórios

```
mro_agenticops/
├── src/                      # Código fonte modular
│   ├── preprocess.py         # Pipeline de ingestão de PDFs
│   ├── train.py              # Geração de embeddings + índice FAISS
│   ├── evaluate.py           # Métricas RAG (MRR, NDCG, recall)
│   ├── retriever.py          # (A criar) Motor de busca vetorial
│   ├── llm.py                # (A criar) Cliente Ollama
│   ├── query.py              # (A criar) Orquestração RAG end-to-end
│   └── agent.py              # (A criar) Lógica agentic futura
├── data/
│   ├── raw/                  # PDFs originais
│   ├── interim/              # Chunks processados (JSON/Parquet)
│   └── processed/            # FAISS index + metadados
├── pipelines/                # Scripts de orquestração
│   ├── ingest_pipeline.py    # Automação preprocess → train
│   └── eval_pipeline.py      # Automação de avaliação contínua
├── prompts/                  # Templates de prompts versionados
│   ├── system_prompt.txt     # Contexto do sistema
│   └── rag_template.txt      # Template para RAG
├── infra/                    # Scripts de setup local
│   ├── setup_ollama.sh       # Install e pull de modelos
│   └── docker-compose.yml    # (Opcional) Ollama containerizado
├── monitoring/               # Observabilidade
│   ├── mlruns/               # MLflow experiments
│   └── logs/                 # Logs estruturados
├── governance/               # Compliance e auditoria
│   └── data_policy.md        # Políticas de uso de dados
├── configs/
│   └── default.yaml          # Configurações centralizadas
├── tests/                    # Testes unitários e integração
├── .github/workflows/        # CI/CD (opcional)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 🚀 Início Rápido

### 1. Configurar Ambiente

```bash
# Criar ambiente conda
conda create -n mro python=3.11 -y
conda activate mro

# Instalar dependências
pip install -r requirements.txt
```

### 2. Configurar Ollama

```bash
# Iniciar servidor (terminal separado)
ollama serve

# Baixar modelo (ex: llama3)
ollama pull llama3
```

### 3. Processar Dados

```bash
# Extração e chunking de PDFs
python src/preprocess.py

# Gerar embeddings e índice FAISS
python src/train.py
```

### 4. Executar RAG

```python
from src.query import RAGPipeline

rag = RAGPipeline(config_path="configs/default.yaml")
resposta = rag.query("O que é o Modelo de Responsabilidade Organizacional?")
print(resposta)
```

---

## 📊 Observability & Evaluation

### Métricas RAG

- **Retrieval Metrics**: MRR@10, Recall@5, NDCG@10
- **Generation Metrics**: BLEU, ROUGE, BERTScore
- **Latência**: p50, p95, p99 para retrieval e geração
- **Contexto**: Tokens consumidos, chunk relevance score

### MLflow Tracking

```python
import mlflow

mlflow.set_experiment("mro_rag_v1")
with mlflow.start_run():
    mlflow.log_param("model", "llama3")
    mlflow.log_metric("recall@5", 0.87)
```

---

## 🔮 Roadmap para Evolução Agentic

### Fase 1: RAG Básico (Atual)
- [x] Ingestão de PDFs
- [x] Embeddings + FAISS
- [ ] Query pipeline end-to-end
- [ ] Avaliação automatizada

### Fase 2: RAG Avançado
- [ ] Reranking com cross-encoder
- [ ] Hybrid search (BM25 + vetorial)
- [ ] Query decomposition
- [ ] Citation tracking

### Fase 3: Agentic RAG
- [ ] ReAct agent com ferramentas
- [ ] Planejamento multi-step
- [ ] Memória de conversação
- [ ] Self-correction loops

### Fase 4: Multi-Agent System
- [ ] Agente "Pesquisador" (retrieval specialist)
- [ ] Agente "Analista" (synthesis)
- [ ] Agente "Validador" (fact-checking)
- [ ] Orquestrador central

---

## ⚙️ Configuração

Edite `configs/default.yaml`:

```yaml
# LLM
ollama:
  base_url: http://localhost:11434
  model: llama3
  temperature: 0.1
  max_tokens: 2048

# Retrieval
retriever:
  top_k: 5
  similarity_threshold: 0.7
  rerank: true

# Embeddings
embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu  # ou 'cuda'

# Chunking
chunking:
  chunk_size: 512
  chunk_overlap: 50
```

---

## 🧪 Testes

```bash
pytest tests/ -v
```

---

## 📝 Contribuição

1. Criar branch: `git checkout -b feature/nome`
2. Implementar com testes
3. Documentar mudanças
4. Pull Request para revisão

---

## 📚 Referências

- [Ollama API Docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [FAISS Documentation](https://faiss.ai/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [AgenticOps Patterns](https://www.anthropic.com/index/claude-2-1-prompting)

---

## 📄 Licença

MIT License - Veja LICENSE para detalhes.
