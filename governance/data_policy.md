# Política de Uso de Dados - MRO AgenticOps

## Escopo
Este documento define as políticas de governança e uso de dados para o projeto mro_agenticops.

## Princípios

### 1. Privacidade Offline-First
- Todos os dados são processados localmente
- Nenhuma informação é enviada para serviços externos
- Modelos LLM executam 100% em ambiente local (Ollama)

### 2. Controle de Acesso
- Documentos PDF originais: acesso restrito à equipe autorizada
- Embeddings e índices: considerados dados derivados, mesma política
- Logs e métricas: acesso para fins de auditoria

### 3. Retenção de Dados
- PDFs raw: retenção indefinida (fontes primárias)
- Dados interim (chunks): retenção por 6 meses
- Índices FAISS: retenção até nova versão
- Logs: rotação a cada 90 dias

### 4. Qualidade e Auditoria
- Rastreabilidade: cada chunk mantém metadados da fonte
- Versionamento: tracking de alterações nos documentos
- Validação: revisão manual de chunks críticos

## Conformidade
- LGPD: Dados fictícios ou anonimizados quando aplicável
- ISO 27001: Controles de segurança local
- Documentação: logs de acesso para auditoria

## Responsabilidades
- **Data Owner**: Equipe MRO
- **Data Steward**: Responsável técnico pelo pipeline
- **Data Users**: Usuários autorizados do sistema RAG

## Revisão
Este documento deve ser revisado semestralmente ou quando houver mudança regulatória.

---
*Última atualização: 2026-02-07*
