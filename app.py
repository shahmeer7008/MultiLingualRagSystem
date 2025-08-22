
import os
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langdetect import detect
import PyPDF2
from docx import Document
import io
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    CHUNK_SIZE = 180
    CHUNK_OVERLAP = 40
    SUPPORTED_LANGUAGES = ['en', 'ur', 'ar', 'fr', 'es', 'de', 'zh', 'hi', 'ja', 'ko', 'it', 'pt', 'ru']

class MultilingualRAG:
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)


        self.embedding_model.max_seq_length = 256
        if hasattr(self.embedding_model, "tokenizer"):
            try:
                self.embedding_model.tokenizer.model_max_length = 256
                
                self.embedding_model.tokenizer.init_kwargs["model_max_length"] = 256
            except Exception:
                pass


        self.index = None
        self.documents = []
        self.metadata = []
        self.groq_client = Groq(api_key=Config.GROQ_API_KEY)
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX file"""
        try:
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            return detect(text[:1000])  
        except:
            return 'en' 
    
    def chunk_text(self, text, chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP):
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
    def process_document(self, file, filename):
        """Process uploaded document"""
        text = ""
        
        if filename.endswith('.pdf'):
            text = self.extract_text_from_pdf(file)
        elif filename.endswith('.docx'):
            text = self.extract_text_from_docx(file)
        elif filename.endswith('.txt'):
            text = str(file.read(), "utf-8")
        else:
            st.error("Unsupported file format. Please upload PDF, DOCX, or TXT files.")
            return
        
        if not text.strip():
            st.error("No text could be extracted from the file.")
            return
        
        
        language = self.detect_language(text)
        
        
        chunks = self.chunk_text(text)
        
        
        for i, chunk in enumerate(chunks):
            self.documents.append(chunk)
            self.metadata.append({
                'filename': filename,
                'chunk_id': i,
                'language': language,
                'total_chunks': len(chunks)
            })
        
        st.success(f"✅ Processed {filename}: {len(chunks)} chunks, Language: {language}")
        
    def build_index(self):
        """Build FAISS index from documents"""
        if not self.documents:
            st.error("No documents to index. Please upload some documents first.")
            return
        
        with st.spinner("Building search index..."):
           
            embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)
            
           
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  
            
           
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
        st.success(f"✅ Index built with {len(self.documents)} chunks!")
    
    def search(self, query, k=5):
        """Search for relevant documents"""
        if self.index is None:
            return []
        
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx]
                })
        
        return results
    
    def generate_response(self, query, context_docs, target_language='auto'):
        """Generate response using Groq"""
        
        context = "\n\n".join([doc['text'] for doc in context_docs])
        
        
        query_lang = self.detect_language(query)
        
       
        language_names = {
            'en': 'English',
            'ur': 'Urdu',
            'ar': 'Arabic', 
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'zh': 'Chinese',
            'hi': 'Hindi',
            'ja': 'Japanese',
            'ko': 'Korean',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian'
        }
        
        
        if target_language == 'auto':
            response_lang = language_names.get(query_lang, 'English')
            lang_code = query_lang
        else:
            response_lang = language_names.get(target_language, 'English')
            lang_code = target_language
        
       
        system_prompt = f"""You are a helpful multilingual assistant. Answer the user's question based on the provided context.

IMPORTANT INSTRUCTIONS:
- You MUST respond in {response_lang} language (language code: {lang_code})
- Even if the question is in a different language, respond in {response_lang}
- Be accurate and concise
- If the context doesn't contain relevant information, say so in {response_lang}
- Translate your answer to {response_lang} if needed
- Use natural {response_lang} expressions and idioms

Context:
{context}

Remember: Your response must be entirely in {response_lang}, regardless of the input language."""
        
        try:
            response = self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {query}\n\nPlease answer this question in {response_lang} based on the provided context."}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def save_state(self):
        """Save RAG state to session"""
        if self.index is not None:
           
            index_bytes = faiss.serialize_index(self.index)
            st.session_state['rag_index'] = index_bytes
            st.session_state['rag_documents'] = self.documents
            st.session_state['rag_metadata'] = self.metadata
    
    def load_state(self):
        """Load RAG state from session"""
        if 'rag_index' in st.session_state:
            index_bytes = st.session_state['rag_index']
            self.index = faiss.deserialize_index(index_bytes)
            self.documents = st.session_state['rag_documents']
            self.metadata = st.session_state['rag_metadata']
            return True
        return False

def main():
    st.set_page_config(
        page_title="Multilingual RAG System",
        page_icon="",
        layout="wide"
    )
    
    st.title("Multilingual RAG System")
    st.markdown("Upload documents in any language and ask questions!")
    
    
    if 'rag' not in st.session_state:
        st.session_state.rag = MultilingualRAG()
        st.session_state.rag.load_state()
    
    rag = st.session_state.rag
    
   
    with st.sidebar:
        st.header("Document Management")
        
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT"
        )
        
       
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    rag.process_document(uploaded_file, uploaded_file.name)
                    rag.save_state()
        
       
        if st.button("Build Search Index", type="primary"):
            rag.build_index()
            rag.save_state()
        
        
        if rag.documents:
            st.subheader("Index Statistics")
            st.metric("Total Chunks", len(rag.documents))
            
           
            languages = {}
            for meta in rag.metadata:
                lang = meta['language']
                languages[lang] = languages.get(lang, 0) + 1
            
            st.write("**Languages:**")
            for lang, count in languages.items():
                st.write(f"- {lang}: {count} chunks")
        
        
        if st.button("Clear All Data"):
            st.session_state.rag = MultilingualRAG()
            if 'rag_index' in st.session_state:
                del st.session_state['rag_index']
            if 'rag_documents' in st.session_state:
                del st.session_state['rag_documents']
            if 'rag_metadata' in st.session_state:
                del st.session_state['rag_metadata']
            st.rerun()
    
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask Questions")
        
      
        query = st.text_input(
            "Enter your question:",
            placeholder="Ask anything about your documents in any language..."
        )
        
       
        language_options = {
            'auto': 'Auto-detect',
            'en': '🇺🇸 English', 
            'ur': '🇵🇰 Urdu',
            'ar': '🇸🇦 Arabic',
            'fr': '🇫🇷 French',
            'es': '🇪🇸 Spanish', 
            'de': '🇩🇪 German',
            'zh': '🇨🇳 Chinese',
            'hi': '🇮🇳 Hindi',
            'ja': '🇯🇵 Japanese',
            'ko': '🇰🇷 Korean',
            'it': '🇮🇹 Italian',
            'pt': '🇵🇹 Portuguese',
            'ru': '🇷🇺 Russian'
        }
        
        target_language = st.selectbox(
            "Response Language:",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            help="Select the language for the AI response - works regardless of your question's language"
        )
        
        
        if st.button("Search & Answer", type="primary") and query:
            if not rag.documents:
                st.error("Please upload and process some documents first!")
            else:
                with st.spinner("Searching and generating response..."):
                  
                    results = rag.search(query, k=5)
                    
                    if results:
                        
                        response = rag.generate_response(query, results, target_language)
                        
                        
                        st.subheader("Response")
                        
                        
                        detected_query_lang = rag.detect_language(query)
                        query_lang_name = {
                            'en': 'English', 'ur': 'Urdu', 'ar': 'Arabic', 'fr': 'French',
                            'es': 'Spanish', 'de': 'German', 'zh': 'Chinese', 'hi': 'Hindi',
                            'ja': 'Japanese', 'ko': 'Korean', 'it': 'Italian', 
                            'pt': 'Portuguese', 'ru': 'Russian'
                        }.get(detected_query_lang, 'Unknown')
                        
                        response_lang_name = language_options[target_language].split(' ', 1)[1] if target_language != 'auto' else query_lang_name
                        
                        if target_language != 'auto':
                            st.info(f"Question detected in: **{query_lang_name}** → Response in: **{response_lang_name}**")
                        else:
                            st.info(f"Auto-detected language: **{query_lang_name}**")
                        
                        st.write(response)
                        
                       
                        with st.expander("Sources"):
                            for i, result in enumerate(results):
                                st.write(f"**Source {i+1}** (Score: {result['score']:.3f})")
                                st.write(f"File: {result['metadata']['filename']}")
                                st.write(f"Language: {result['metadata']['language']}")
                                st.write(f"Chunk: {result['metadata']['chunk_id']+1}/{result['metadata']['total_chunks']}")
                                st.text_area(f"Content {i+1}", result['text'], height=100, key=f"source_{i}")
                                st.divider()
                    else:
                        st.warning("No relevant documents found for your query.")
    
    with col2:
        st.header("About")
        st.markdown("""
        **Features:**
        - Supports 13+ languages
        - PDF, DOCX, TXT files
        - Semantic search
        - AI-powered responses
        - Session persistence
        
        **Supported Languages:**
        English, Urdu, Arabic, French, Spanish, German, Chinese, Hindi, Japanese, Korean, Italian, Portuguese, Russian
        
        **How to use:**
        1. Upload documents in sidebar
        2. Click "Process" for each file
        3. Build search index
        4. Ask questions!
        """)
        
        
        st.subheader("Sample Questions")
        sample_questions = [
            "What is this document about?",
            "Summarize the main points",
            "اہم نکات کیا ہیں؟", 
            "¿Cuáles son los puntos principales?", 
            "Quels sont les points principaux?",  
        ]
        
        for sq in sample_questions:
            if st.button(sq, key=f"sample_{sq}"):
                st.session_state['sample_query'] = sq

if __name__ == "__main__":
    main()