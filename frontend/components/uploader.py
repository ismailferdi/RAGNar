import streamlit as st
import httpx
import os
from json import loads, JSONDecodeError

API_BASE_URL = st.secrets["RAGNAR_API_URL"]


def _parse_documents(data):
    if data is None:
        return []
    if isinstance(data, str):
        try:
            data = loads(data)
        except JSONDecodeError:
            st.error(f"Backend returned invalid JSON string: {data[:200]}")
            return []
    if isinstance(data, dict):
        for key in ("documents", "data", "results", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            if "id" in data:
                data = [data]
            else:
                st.error(
                    f"Backend returned dict with unexpected keys: {list(data.keys())}"
                )
                return []
    if not isinstance(data, list):
        st.error(f"Expected list from backend, got {type(data).__name__}")
        return []
    parsed = []
    for item in data:
        if isinstance(item, str):
            try:
                item = loads(item)
            except JSONDecodeError:
                st.error(f"Skipping invalid document string: {item[:200]}")
                continue
        if isinstance(item, dict):
            parsed.append(item)
        else:
            st.error(f"Skipping unexpected document type: {type(item).__name__}")
    return parsed


def _do_delete(doc_id):
    try:
        response = httpx.delete(
            f"{API_BASE_URL}/documents/{doc_id}/", follow_redirects=True
        )
        response.raise_for_status()
        # Remove from local state immediately so the UI updates even if refresh fails
        docs = st.session_state.get("documents", [])
        st.session_state["documents"] = [
            doc
            for doc in docs
            if isinstance(doc, dict) and str(doc.get("id")) != str(doc_id)
        ]
        _refresh_documents()
        st.success("Document deleted successfully.")
    except httpx.RequestError as e:
        st.error(f"Network error while deleting: {e}")
    except httpx.HTTPStatusError as e:
        st.error(
            f"Failed to delete document: HTTP {e.response.status_code} - {e.response.text}"
        )


def _refresh_documents():
    try:
        response = httpx.get(f"{API_BASE_URL}/documents/", follow_redirects=True)
        response.raise_for_status()
        data = response.json()
        st.session_state["documents"] = _parse_documents(data)
    except httpx.RequestError as e:
        st.warning(f"Failed to refresh document list: {e}")
    except JSONDecodeError as e:
        st.warning(f"Backend returned invalid JSON: {e}")
    except httpx.HTTPStatusError as e:
        st.warning(f"Backend error {e.response.status_code}: {e.response.text}")


def _handle_conflict(files):
    st.warning("This document has already ingested.")
    force = st.checkbox("Force re-ingest?", key="force_reingest")
    if force:
        try:
            with st.spinner("Re-ingesting document..."):
                response = httpx.post(
                    f"{API_BASE_URL}/ingest/?force=true",
                    files=files,
                    timeout=60.0,
                    follow_redirects=True,
                )
            if response.status_code == 200:
                data = response.json()
                st.success(
                    f"Re-ingested {data['chunk_count']} chunks from {data['sanitized_filename']}"
                )
            else:
                err = response.text or f"HTTP {response.status_code}"
                st.error(f"Re-ingestion failed: {err}")
        except httpx.RequestError as e:
            st.error(f"Network error: {e}")
        finally:
            st.session_state["force_reingest"] = False


def render():
    with st.sidebar:
        st.title("Uploaded Documents")

        # Execute pending delete before drawing any widgets
        pending_id = st.session_state.get("_pending_delete_id")
        if pending_id is not None:
            st.session_state["_pending_delete_id"] = None
            _do_delete(pending_id)

        uploaded_file = st.file_uploader(
            "Upload a document", type=["pdf", "txt"], accept_multiple_files=False
        )
        if uploaded_file is not None:
            file_id = f"uploaded_{uploaded_file.name}-{uploaded_file.size}"
            if st.session_state.get("last_uploaded_id") != file_id:
                st.session_state["last_uploaded_id"] = file_id
                files = {
                    "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                }
                try:
                    with st.spinner("Ingesting document..."):
                        response = httpx.post(
                            f"{API_BASE_URL}/ingest/",
                            files=files,
                            timeout=60.0,
                            follow_redirects=True,
                        )
                    if response.status_code == 200:
                        data = response.json()
                        st.success(
                            f"Ingested {data['chunk_count']} chunks from {data['sanitized_filename']}"
                        )
                        _refresh_documents()
                    elif response.status_code == 409:
                        _handle_conflict(files)
                    elif response.status_code == 422:
                        st.error(
                            "This pdf appears to contain scanned images rather than text "
                            "and cannot be processed."
                        )
                    else:
                        err = response.text or f"HTTP {response.status_code}"
                        st.error(f"Ingestion failed: {err}")
                except httpx.RequestError as e:
                    st.error(f"Network error: {e}")

        st.divider()
        st.subheader("Ingested Documents")

        documents = _parse_documents(st.session_state.get("documents"))

        if not documents:
            st.info("No documents uploaded yet.")

        for document in documents:
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(document.get("sanitized_filename", "Unknown"))
                with col2:
                    doc_id = document.get("id")
                    if doc_id is not None:
                        if st.button("Delete", key=f"delete_{doc_id}"):
                            st.session_state["_pending_delete_id"] = doc_id
                            st.rerun()
