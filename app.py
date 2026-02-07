"""Streamlit UI for MRO AgenticOps RAG system."""
from pathlib import Path
import streamlit as st

from src.query import RAGPipeline


APP_TITLE = "MRO AgenticOps - RAG Local"


def check_prereqs() -> list[str]:
    """Check if required artifacts exist and return warnings."""
    warnings = []
    if not Path("data/raw").exists():
        warnings.append("Pasta data/raw nao encontrada.")
    if not Path("data/processed/faiss.index").exists():
        warnings.append("Indice FAISS nao encontrado. Execute o pipeline primeiro.")
    return warnings


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("RAG offline com Ollama + FAISS")

    warnings = check_prereqs()
    if warnings:
        for w in warnings:
            st.warning(w)

    with st.sidebar:
        st.header("Configuracao")
        config_path = st.text_input("Config", "configs/default.yaml")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
        show_sources = st.checkbox("Mostrar fontes", value=True)
        stream = st.checkbox("Streaming", value=False)

    question = st.text_area("Pergunta", "O que e o Modelo de Responsabilidade Organizacional?")

    if st.button("Consultar"):
        if not question.strip():
            st.error("Digite uma pergunta.")
            return

        with st.spinner("Consultando..."):
            rag = RAGPipeline(config_path=config_path)
            result = rag.query(question, top_k=top_k, stream=stream)

        st.subheader("Resposta")
        if stream and hasattr(result["answer"], "__iter__"):
            placeholder = st.empty()
            buffer = ""
            for chunk in result["answer"]:
                buffer += chunk
                placeholder.markdown(buffer)
        else:
            st.write(result["answer"])

        if show_sources:
            st.subheader("Fontes")
            if result["sources"]:
                for i, src in enumerate(result["sources"], 1):
                    meta = src.get("metadata", {})
                    st.markdown(
                        f"**{i}.** {meta.get('source', 'desconhecido')} | "
                        f"score={src.get('score', 0):.3f}"
                    )
                    st.write(src.get("text", ""))
            else:
                st.info("Nenhuma fonte encontrada para esta pergunta.")


if __name__ == "__main__":
    main()
