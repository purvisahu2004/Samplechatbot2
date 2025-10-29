import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader

# --- Configuration ---
GEMINI_API_KEY = "AIzaSyBnNcI3bPJWgdbbTGBqqRr3tM9hSWYvWaQ"   # 🔒 Replace with your Gemini API key
PDF_PATH = "ai-terms-cheatsheet.pdf"                 # 📄 Replace with your local PDF

# --- Streamlit setup ---
st.set_page_config(page_title="📄 Gemini PDF Q&A (Agentic Chunking)", page_icon="🤖")
st.title("📄PDF Q&A Chatbot ")

# --- Configure Gemini ---
genai.configure(api_key=GEMINI_API_KEY)

# --- Load PDF ---
st.info(f"📂 Loading PDF: {PDF_PATH}")
pdf_reader = PdfReader(PDF_PATH)
text = ""
for page in pdf_reader.pages:
    text += page.extract_text()

# --- Custom Agentic Chunking Function ---
def agentic_chunking(text, model_name="gemini-2.5-flash-lite"):
    """
    Uses Gemini to decide semantically meaningful chunk boundaries.
    """
    model = genai.GenerativeModel(model_name)
    prompt = (
        "Split the following text into meaningful, self-contained sections "
        "(each about 500–1000 words) based on topic or logical boundaries. "
        "Return each chunk separated by a line containing only '---CHUNK---'.\n\n"
        f"{text}"
    )
    response = model.generate_content(prompt)
    chunks = response.text.split("---CHUNK---")
    return [c.strip() for c in chunks if c.strip()]

# --- Perform Agentic Chunking ---
st.info("🧩 Performing AI-based (Agentic) Chunking...")
chunks = agentic_chunking(text)
st.success(f"✅ PDF split into {len(chunks)} intelligent chunks by Gemini.")

# --- User Question ---
st.subheader("💬 Ask a Question about AI AND ML")
user_question = st.text_input("Type your question here")

model = genai.GenerativeModel("gemini-2.5-flash-lite")

if st.button("Get Answer"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing relevant chunks..."):
            # Keyword match to find relevant chunks
            relevant_chunks = [chunk for chunk in chunks if any(word.lower() in chunk.lower() for word in user_question.split())]
            if not relevant_chunks:
                relevant_chunks = chunks[:3]

            context = "\n\n".join(relevant_chunks[:5])

            prompt = f"""
            You are a helpful assistant. 
            Answer the question based only on the context below.

            Context:
            {context}

            Question: {user_question}
            """
            response = model.generate_content(prompt)

        st.write("### 🧠 Answer:")
        st.write(response.text)
