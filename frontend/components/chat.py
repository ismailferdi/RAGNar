import streamlit as st
import httpx

from .source_viewer import render_sources

API_BASE_URL = st.secrets["RAGNAR_API_URL"]


def render():

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if not message.get("grounded", True):
                    st.info(
                        "This answer was generated without any matching context — "
                        "the document may not contain information relevant to your question."
                    )
                sources = message.get("sources", [])
                if sources:
                    with st.expander("View sources"):
                        render_sources(sources)

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = httpx.post(
                        f"{API_BASE_URL}/ask/",
                        json={"question": prompt},
                        follow_redirects=True,
                        timeout=60.0,
                    )
                    response.raise_for_status()
                    answer_data = response.json()
                except httpx.RequestError as e:
                    st.error(f"An error occurred while requesting the backend: {e}")
                    return
                except httpx.HTTPStatusError as e:
                    st.error(
                        f"Backend returned an error: {e.response.status_code} - {e.response.text}"
                    )
                    return

            answer = answer_data.get("answer", "No answer provided.")
            sources = answer_data.get("sources", [])
            grounded = answer_data.get("grounded", False)

            st.markdown(answer)
            if not grounded:
                st.info(
                    "This answer was generated without any matching context — "
                    "the document may not contain information relevant to your question."
                )

            if sources:
                with st.expander("View sources"):
                    render_sources(sources)

            st.session_state["chat_history"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "grounded": grounded,
                }
            )
