# DVC Configuration Guide

## 📦 DVC Setup

Este projeto usa **DVC (Data Version Control)** para versionamento de dados e artefatos de ML.

### ✅ Status Atual

- ✅ DVC inicializado
- ✅ Data raw (`data/raw/`) rastreado pelo DVC
- ✅ Remote storage configurado: `C:\Programas\MRO_RAG\dvc_storage`
- ✅ Push inicial completo

---

## 🔄 Workflow DVC

### Adicionar Novos Dados

```bash
# Adicionar novo arquivo/diretório ao DVC
dvc add data/raw/novo_documento.pdf

# Commitar mudanças
git add data/raw.dvc data/.gitignore
git commit -m "Add novo_documento.pdf"

# Push para DVC remote
dvc push

# Push para GitHub
git push
```

### Baixar Dados (Clone Novo)

```bash
# Clonar repositório
git clone https://github.com/gutpassos/mro_agenticops.git
cd mro_agenticops

# Baixar dados do DVC
dvc pull
```

### Atualizar Dados

```bash
# Alguém adicionou novos dados no remote
git pull
dvc pull  # Sincroniza dados
```

---

## 🌐 Configurar Remote na Nuvem (Opcional)

### Google Drive

```bash
dvc remote add -d gdrive gdrive://1aB2c3D4e5F6g7H8i9J0k
dvc push
```

### AWS S3

```bash
dvc remote add -d s3remote s3://meu-bucket/mro-data
dvc remote modify s3remote region us-east-1
dvc push
```

### Azure Blob Storage

```bash
dvc remote add -d azure azure://mycontainer/path
dvc push
```

---

## 📊 Arquivos DVC

- **`data/raw.dvc`**: Metadado dos PDFs rastreados
- **`data/.gitignore`**: Ignora dados reais (apenas metadados vão pro git)
- **`.dvc/config`**: Configuração de remotes
- **`.dvc/.gitignore`**: Cache local do DVC

---

## 🔍 Comandos Úteis

```bash
# Ver status de arquivos rastreados
dvc status

# Ver diferenças de dados
dvc diff

# Ver pipeline de dependências (quando criado)
dvc dag

# Listar remotes
dvc remote list

# Verificar integridade
dvc check
```

---

## 🎯 Benefícios

✅ **Versionamento de dados** como código  
✅ **Colaboração** sem duplicar grandes arquivos  
✅ **Reprodutibilidade** de experimentos  
✅ **Storage eficiente** com deduplicação  
✅ **Git não fica pesado** (apenas metadados)

---

## 📝 Próximos Passos

1. **Criar pipeline DVC** para processar dados:
   ```bash
   dvc run -n preprocess \
     -d data/raw \
     -o data/interim/chunks.json \
     python src/preprocess.py
   ```

2. **Adicionar stage de treinamento**:
   ```bash
   dvc run -n train \
     -d data/interim/chunks.json \
     -o data/processed/faiss.index \
     python src/train.py
   ```

3. **Rastrear métricas**:
   ```bash
   dvc metrics show
   dvc plots show
   ```

---

## 🆘 Troubleshooting

### Erro: "dvc.lock corrupted"
```bash
dvc checkout --relink
```

### Cache muito grande
```bash
# Limpar cache antigo
dvc gc --workspace
```

### Conflitos de merge
```bash
dvc resolve
```

---

## 📚 Documentação

- [DVC Docs](https://dvc.org/doc)
- [DVC Tutorial](https://dvc.org/doc/start)
- [DVC with Git](https://dvc.org/doc/use-cases/versioning-data-and-model-files)
