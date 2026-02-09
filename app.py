import streamlit as st
import os
import time
import hashlib
import tempfile
import numpy as np
from typing import List, Dict
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from pinecone import Pinecone, ServerlessSpec
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter


st.set_page_config(
    page_title="Exam Assistant",
    page_icon="📖",
    layout="wide"
)

# Session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.messages = []
    st.session_state.user_id = None
    st.session_state.notes_processed = False
    st.session_state.books_processed = False
    st.session_state.metrics = {
        "total_queries": 0,
        "stage1_used": 0,
        "stage2_used": 0,
        "avg_response_time": [],
        "avg_confidence": []
    }

EMBED_DIM = 384


# ---- Sidebar ----

with st.sidebar:
    st.header("Configuration")

    pinecone_key = os.getenv("pineconeapikey") or st.text_input(
        "Pinecone API Key",
        type="password",
        help="Get free at pinecone.io"
    )
    groq_key = os.getenv("myfirstapikey") or st.text_input(
        "Groq API Key",
        type="password",
        help="Get free at console.groq.com"
    )

    st.divider()

    st.subheader("How It Works")
    st.markdown("""
    **Stage 1:** Notes & Sample Papers
    - Fast, focused retrieval
    - Priority-weighted scoring

    **Stage 2:** Reference Books
    - Detailed retrieval
    - Used when Stage 1 confidence is low

    Intelligent routing picks the best stage automatically.
    """)

    st.divider()

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.3, max_value=0.9, value=0.5, step=0.05,
        help="Controls when to fall back from Stage 1 to Stage 2"
    )

    temperature = st.slider(
        "Response Temperature",
        min_value=0.0, max_value=1.0, value=0.2, step=0.1,
        help="Lower = more factual"
    )

    st.divider()

    st.subheader("Upload Status")
    if st.session_state.notes_processed:
        st.success("Notes processed")
    else:
        st.warning("No notes uploaded yet")

    if st.session_state.books_processed:
        st.success("Reference books processed")
    else:
        st.warning("No books uploaded yet")

    st.divider()

    if st.session_state.metrics["total_queries"] > 0:
        st.subheader("Performance")
        m = st.session_state.metrics
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Queries", m["total_queries"])
            st.metric("Stage 1", f"{m['stage1_used']}")
        with c2:
            if m["avg_response_time"]:
                st.metric("Avg Time", f"{np.mean(m['avg_response_time']):.2f}s")
            st.metric("Stage 2", f"{m['stage2_used']}")

    st.divider()

    st.subheader("About")
    st.markdown("""
    Built by **Harsh Varia**

    [GitHub](https://github.com/HarshVaria) | [LinkedIn](https://linkedin.com/in/HarshVaria)
    """)


# ---- Load models ----

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embedding_model, reranker


# ---- Pinecone setup ----

def initialize_pinecone(api_key, user_id):
    pc = Pinecone(api_key=api_key)

    notes_index_name = f"exam-notes-{user_id}"
    books_index_name = f"exam-books-{user_id}"

    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if notes_index_name not in existing_indexes:
        pc.create_index(
            name=notes_index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)

    if books_index_name not in existing_indexes:
        pc.create_index(
            name=books_index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(5)

    return pc.Index(notes_index_name), pc.Index(books_index_name)


# ---- PDF text extraction ----

def extract_text_from_pdf(pdf_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name

        reader = PdfReader(tmp_path)
        text = ""

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            text += f"\n[page {page_num + 1}]\n{page_text}"

        os.unlink(tmp_path)

        if not text.strip():
            return None

        return text

    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None


# ---- Chunking ----

def smart_chunk_notes(text: str, chunk_size: int = 600, chunk_overlap: int = 100) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\nQuestion",
            "\nQ.",
            "\n\n---\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )
    return text_splitter.split_text(text)


def semantic_text_splitter(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)


def calculate_chunk_priority(chunk_text: str) -> float:
    priority = 1.0
    text_lower = chunk_text.lower()

    if any(marker in text_lower for marker in ['question', 'q.', 'q1', 'problem']):
        priority *= 1.5

    if 'marks' in text_lower or 'points' in text_lower:
        priority *= 1.3

    important_keywords = ['important', 'note:', 'remember', 'key concept', 'definition']
    if any(keyword in text_lower for keyword in important_keywords):
        priority *= 1.2

    if any(marker in text_lower for marker in ['answer', 'solution', 'ans.']):
        priority *= 1.4

    return priority


# ---- Process and upload ----

def process_notes(pdf_files, index, embedding_model):
    all_chunks = []
    progress_bar = st.progress(0)
    status = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        status.text(f"Reading {pdf_file.name}...")
        text = extract_text_from_pdf(pdf_file)

        if text:
            chunks = smart_chunk_notes(text, chunk_size=600, chunk_overlap=100)

            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": pdf_file.name,
                    "chunk_id": f"{pdf_file.name}_note_{j}",
                    "category": "notes",
                    "priority": calculate_chunk_priority(chunk),
                    "char_count": len(chunk)
                })

        progress_bar.progress((i + 1) / len(pdf_files))

    if not all_chunks:
        st.error("No text extracted from PDFs")
        progress_bar.empty()
        status.empty()
        return False

    # Priority distribution
    high_p = sum(1 for c in all_chunks if c["priority"] >= 1.5)
    med_p = sum(1 for c in all_chunks if 1.2 <= c["priority"] < 1.5)
    low_p = sum(1 for c in all_chunks if c["priority"] < 1.2)
    st.info(f"Priority: High={high_p}, Medium={med_p}, Base={low_p}")

    status.text(f"Uploading {len(all_chunks)} notes chunks...")

    BATCH_SIZE = 100
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embedding_model.encode(texts, show_progress_bar=False)

        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vectors.append((
                chunk["chunk_id"],
                embedding.tolist(),
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "category": "notes",
                    "priority": chunk["priority"],
                    "char_count": chunk["char_count"]
                }
            ))

        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            st.error(f"Error upserting notes batch: {e}")

    status.text(f"Done. {len(all_chunks)} notes chunks processed.")
    time.sleep(1)
    status.empty()
    progress_bar.empty()
    return True


def process_books(pdf_files, index, embedding_model):
    all_chunks = []
    progress_bar = st.progress(0)
    status = st.empty()

    for i, pdf_file in enumerate(pdf_files):
        status.text(f"Reading {pdf_file.name}...")
        text = extract_text_from_pdf(pdf_file)

        if text:
            chunks = semantic_text_splitter(text, chunk_size=1000, chunk_overlap=200)

            for j, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": pdf_file.name,
                    "chunk_id": f"{pdf_file.name}_chunk_{j}",
                    "category": "reference_book",
                    "char_count": len(chunk)
                })

        progress_bar.progress((i + 1) / len(pdf_files))

    if not all_chunks:
        st.error("No text extracted from PDFs")
        progress_bar.empty()
        status.empty()
        return False

    status.text(f"Uploading {len(all_chunks)} reference chunks...")

    BATCH_SIZE = 100
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embedding_model.encode(texts, show_progress_bar=False)

        vectors = []
        for chunk, embedding in zip(batch, embeddings):
            vectors.append((
                chunk["chunk_id"],
                embedding.tolist(),
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "category": "reference_book"
                }
            ))

        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            st.error(f"Error upserting books batch: {e}")

    status.text(f"Done. {len(all_chunks)} reference chunks processed.")
    time.sleep(1)
    status.empty()
    progress_bar.empty()
    return True


# ---- Two stage retrieval ----

def stage1_retrieve(question, notes_index, embedding_model, reranker, top_k=5, rerank_top_k=3):
    try:
        query_embedding = embedding_model.encode(question).tolist()
        results = notes_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        matches = results.get("matches", [])

        if matches:
            pairs = [[question, m["metadata"]["text"]] for m in matches]
            rerank_scores = reranker.predict(pairs)

            for match, score in zip(matches, rerank_scores):
                priority = match["metadata"].get("priority", 1.0)
                match["confidence"] = float(score) * priority

            matches.sort(key=lambda x: x["confidence"], reverse=True)
            return matches[:rerank_top_k]

        return []

    except Exception as e:
        st.error(f"Stage 1 error: {e}")
        return []


def stage2_retrieve(question, books_index, embedding_model, reranker, top_k=4):
    try:
        query_embedding = embedding_model.encode(question).tolist()
        results = books_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        matches = results.get("matches", [])

        if matches:
            pairs = [[question, m["metadata"]["text"]] for m in matches]
            rerank_scores = reranker.predict(pairs)

            for match, score in zip(matches, rerank_scores):
                match["relevance"] = float(score)
                match["confidence"] = float(score)

            matches.sort(key=lambda x: x["relevance"], reverse=True)

        return matches

    except Exception as e:
        st.error(f"Stage 2 error: {e}")
        return []


def intelligent_routing(question, notes_index, books_index, embedding_model, reranker,
                        confidence_threshold=0.5, notes_available=True, books_available=True):

    stage1_matches = []
    stage2_matches = []

    if notes_available:
        stage1_matches = stage1_retrieve(question, notes_index, embedding_model, reranker, top_k=5, rerank_top_k=3)

    if stage1_matches and stage1_matches[0]["confidence"] > confidence_threshold:
        return {
            "stage": 1,
            "source": "notes_sample_papers",
            "matches": stage1_matches,
            "reason": f"High confidence match (score: {stage1_matches[0]['confidence']:.3f})"
        }
    else:
        if books_available:
            stage2_matches = stage2_retrieve(question, books_index, embedding_model, reranker, top_k=4)

        if stage2_matches:
            return {
                "stage": 2,
                "source": "reference_books",
                "matches": stage2_matches,
                "stage1_matches": stage1_matches,
                "reason": "No high-confidence match in notes, searching reference books"
            }
        elif stage1_matches:
            return {
                "stage": 1,
                "source": "notes_sample_papers",
                "matches": stage1_matches,
                "reason": f"Best available notes match (score: {stage1_matches[0]['confidence']:.3f})"
            }
        else:
            return {
                "stage": 0,
                "source": "none",
                "matches": [],
                "reason": "No relevant information found"
            }


# ---- Answer generation ----

def generate_answer(question, result, groq_client, temp):
    matches = result.get("matches", [])

    if not matches:
        return "No relevant information found in your uploaded materials."

    stage = result["stage"]

    if stage == 1:
        context_parts = []
        for i, match in enumerate(matches[:2], 1):
            text = match["metadata"]["text"]
            source = match["metadata"]["source"]
            confidence = match["confidence"]
            context_parts.append(f"[Source {i}: {source} (confidence: {confidence:.3f})]\n{text}")

        context = "\n\n".join(context_parts)

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an exam preparation assistant. Provide concise, focused answers based on the notes/sample papers provided. Be direct and exam-oriented."
                },
                {
                    "role": "user",
                    "content": f"Notes/Sample Paper Content:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=temp,
            max_tokens=400
        )

    else:
        context = "\n\n".join([
            f"[Source: {m['metadata']['source']}]\n{m['metadata']['text']}"
            for m in matches
        ])

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an exam preparation assistant. Provide detailed, accurate answers based on the reference material provided."
                },
                {
                    "role": "user",
                    "content": f"Reference Material:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=temp + 0.1,
            max_tokens=500
        )

    return response.choices[0].message.content


# ---- Main app ----

st.markdown("## Exam Assistant")
st.markdown("Upload your study materials and get intelligent answers")

if not pinecone_key or not groq_key:
    st.warning("Please enter your API keys in the sidebar to start.")
    st.info("""
    **Get free API keys:**
    - **Pinecone:** [pinecone.io](https://www.pinecone.io/)
    - **Groq:** [console.groq.com](https://console.groq.com/)
    """)
    st.stop()

if not st.session_state.user_id:
    st.session_state.user_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

if not st.session_state.initialized:
    with st.spinner("Loading models..."):
        try:
            st.session_state.embedding_model, st.session_state.reranker = load_models()
            st.session_state.notes_index, st.session_state.books_index = initialize_pinecone(
                pinecone_key, st.session_state.user_id
            )
            st.session_state.groq_client = Groq(api_key=groq_key)
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Initialization error: {e}")
            st.stop()


# Upload section

st.markdown("### Upload Study Materials")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Notes & Sample Papers**")
    st.caption("Used for Stage 1 retrieval (600 char chunks, priority scoring)")

    notes_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        key="notes_uploader"
    )

    if notes_files and st.button("Process Notes", type="primary", key="process_notes"):
        with st.spinner("Processing notes..."):
            success = process_notes(
                notes_files,
                st.session_state.notes_index,
                st.session_state.embedding_model
            )
            if success:
                st.session_state.notes_processed = True
                st.success(f"Processed {len(notes_files)} note file(s)")
                st.rerun()

with col2:
    st.markdown("**Reference Books**")
    st.caption("Used for Stage 2 retrieval (1000 char chunks, detailed)")

    books_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        key="books_uploader"
    )

    if books_files and st.button("Process Books", type="primary", key="process_books"):
        with st.spinner("Processing books..."):
            success = process_books(
                books_files,
                st.session_state.books_index,
                st.session_state.embedding_model
            )
            if success:
                st.session_state.books_processed = True
                st.success(f"Processed {len(books_files)} book file(s)")
                st.rerun()

st.divider()


# Chat section

st.markdown("### Ask Questions")

if not st.session_state.notes_processed and not st.session_state.books_processed:
    st.info("Upload study materials above to start asking questions.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "metadata" in message:
                with st.expander("Details", expanded=False):
                    meta = message["metadata"]

                    if meta["stage"] == 1:
                        st.markdown("**Stage 1: Notes/Sample Papers**")
                    else:
                        st.markdown("**Stage 2: Reference Books**")

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Response Time", f"{meta['response_time']:.2f}s")
                    with c2:
                        st.metric("Confidence", f"{meta['confidence']:.3f}")
                    with c3:
                        st.metric("Sources", len(meta['sources']))

                    st.markdown(f"**Reason:** {meta['reason']}")

                    st.markdown("**Sources:**")
                    for i, source in enumerate(meta["sources"], 1):
                        st.text(f"  {i}. {source}")

    # Chat input
    if prompt := st.chat_input("Ask a question about your study materials..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()

                result = intelligent_routing(
                    question=prompt,
                    notes_index=st.session_state.notes_index,
                    books_index=st.session_state.books_index,
                    embedding_model=st.session_state.embedding_model,
                    reranker=st.session_state.reranker,
                    confidence_threshold=confidence_threshold,
                    notes_available=st.session_state.notes_processed,
                    books_available=st.session_state.books_processed
                )

                # Metrics
                st.session_state.metrics["total_queries"] += 1
                if result["stage"] == 1:
                    st.session_state.metrics["stage1_used"] += 1
                    if result["matches"]:
                        st.session_state.metrics["avg_confidence"].append(result["matches"][0]["confidence"])
                elif result["stage"] == 2:
                    st.session_state.metrics["stage2_used"] += 1

                # Generate
                if result["matches"]:
                    try:
                        answer = generate_answer(prompt, result, st.session_state.groq_client, temperature)
                    except Exception as e:
                        answer = f"Error: {e}"
                else:
                    answer = "No relevant information found. Try rephrasing or uploading more materials."

                elapsed = time.time() - start_time
                st.session_state.metrics["avg_response_time"].append(elapsed)

                # Metadata
                top_conf = result["matches"][0].get("confidence", 0) if result["matches"] else 0

                sources_list = []
                if result["matches"]:
                    for m in result["matches"]:
                        conf = m.get("confidence", m.get("relevance", 0))
                        sources_list.append(f"{m['metadata']['source']} (score: {conf:.3f})")

                metadata = {
                    "stage": result["stage"],
                    "response_time": elapsed,
                    "confidence": top_conf,
                    "sources": sources_list,
                    "reason": result["reason"]
                }

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata
                })

                st.rerun()


# Footer

st.divider()
st.markdown(
    "Built by Harsh Varia | "
    "[GitHub](https://github.com/HarshVaria) | "
    "[LinkedIn](https://linkedin.com/in/HarshVaria)"
)