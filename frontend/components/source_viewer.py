import streamlit as st


def render_sources(sources: list[dict]):
    for source in sources:
        st.markdown(
            f"**{source.get('source_file', 'Unknown')}** - chunk {source.get('chunk_index', '?')}"
        )
        st.text_area(
            "Chunk text", value=source.get("text", ""), height=100, disabled=True
        )
        st.progress(min(source.get("similarity_score", 0.0), 1.0))
        st.divider()
