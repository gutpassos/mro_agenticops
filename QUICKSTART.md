# MRO AgenticOps - Quick Start Guide

## 🚀 Início Rápido (5 minutos)

### 1. Verificar Status
```bash
python setup.py --check
```

### 2. Instalar Dependências
```bash
# Opção 1: Com setup.py
python setup.py --setup

# Opção 2: Direto com pip
pip install -r requirements.txt
```

### 3. Iniciar Ollama (em terminal separado)
```bash
ollama serve
```

### 4. Baixar Modelo LLM
```bash
ollama pull llama3
```

### 5. Processar PDFs e Criar Índice
```bash
python setup.py --pipeline
```

### 6. Fazer Perguntas
```bash
python setup.py --query
```

---

## 📖 Uso Detalhado

### Modo Interativo
```bash
python setup.py --query
```
Digite suas perguntas e pressione Enter. Digite `exit` para sair.

### Uso Programático
```python
from src.query import RAGPipeline

rag = RAGPipeline()
result = rag.query("O que é o Modelo de Responsabilidade Organizacional?")

print(result['answer'])
```

### Executar Testes
```bash
# Todos os testes
pytest

# Apenas testes unitários (sem integração)
pytest -m "not integration"

# Testes específicos
pytest tests/test_preprocess.py
```

---

## 🔧 Configuração Avançada

### Ajustar Parâmetros de Retrieval
Edite `configs/default.yaml`:
```yaml
retriever:
  top_k: 10              # Mais contexto
  similarity_threshold: 0.6  # Menos restritivo
```

### Mudar Modelo LLM
```yaml
ollama:
  model: mistral  # ou mixtral, codellama, etc
```

### Ajustar Chunking
```yaml
chunking:
  chunk_size: 1024  # Chunks maiores
  chunk_overlap: 100
```

---

## 📊 Pipelines

### Pipeline de Ingestão
```bash
python pipelines/ingest_pipeline.py
```
Executa: Preprocess → Train (embeddings + FAISS)

### Pipeline de Avaliação
```bash
python pipelines/eval_pipeline.py
```
Executa métricas de qualidade do RAG

---

## 🧪 Desenvolvimento

### Adicionar Novo PDF
1. Coloque o PDF em `data/raw/`
2. Execute: `python setup.py --pipeline`

### Reindexar Tudo
```bash
# Apagar índice antigo
rm -rf data/interim/* data/processed/*

# Reprocessar
python setup.py --pipeline
```

### Debugging
```python
# Ativar logs detalhados
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🐛 Troubleshooting

### Erro: "Ollama server not running"
**Solução**: Inicie o servidor
```bash
ollama serve
```

### Erro: "FAISS index not found"
**Solução**: Execute o pipeline
```bash
python setup.py --pipeline
```

### Erro: "No PDF files found"
**Solução**: Adicione PDFs em `data/raw/`

### Erro: ModuleNotFoundError
**Solução**: Instale dependências
```bash
python setup.py --setup
```

---

## 📚 Exemplos de Perguntas

- "O que é o Modelo de Responsabilidade Organizacional?"
- "Quais são os pilares fundamentais do MRO?"
- "Como o MRO se relaciona com governança corporativa?"
- "Explique o conceito de accountability no contexto do MRO"

---

## 🔄 Workflow Completo

```
1. PDFs → data/raw/
2. python setup.py --pipeline
   ├─ preprocess.py → chunks em data/interim/
   └─ train.py → embeddings + FAISS em data/processed/
3. python setup.py --query
   └─ RAGPipeline → Ollama → Resposta
```

---

## 📞 Suporte

- Documentação completa: [README.md](README.md)
- Arquitetura: Ver diagrama no README
- Issues: Criar issue no repositório
