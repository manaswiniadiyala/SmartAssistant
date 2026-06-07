import streamlit as st
from retriever import answer

st.set_page_config(page_title="College FAQ Bot", page_icon="🎓", layout="centered")

st.title("🎓 College FAQ Bot")
st.caption("Ask anything about exam dates, academic rules, admissions, and more.")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.write(f"- **{s['doc_name']}** — Page {s['page']}")

if query := st.chat_input("Ask a question..."):
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching college documents..."):
            result = answer(query)

        if result["confidence"] == "low":
            st.warning(result["answer"])
        else:
            st.write(result["answer"])
            if result["sources"]:
                with st.expander("📄 Sources"):
                    for s in result["sources"]:
                        st.write(f"- **{s['doc_name']}** — Page {s['page']}")

    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", [])
    })