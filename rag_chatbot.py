import os
import pypdf
import google.generativeai as genai
from sentence_transformers import CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class ClassBasedRAGChatbot:
    """
    A unified, class-based RAG chatbot that implements the complete RAG pipeline:
    1. Ingestion: PDF parsing (pypdf) and chunking (RecursiveCharacterTextSplitter)
    2. Embeddings: sentence-transformers/all-MiniLM-L6-v2 via HuggingFaceEmbeddings
    3. Storage: Persistent Chroma Vector DB (with duplicate prevention)
    4. Two-Stage Retrieval: Initial similarity search followed by CrossEncoder reranking
    5. Generation: Gemini API response generation grounded on retrieved context
    """

    def __init__(
        self,
        db_dir="./rag_db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_model="ms-marco-MiniLM-L-12-v2",
        llm_model="gemini-2.5-flash"
    ):
        """
        Initializes the RAG Chatbot with embedding models, vector database,
        cross-encoder reranker, and LLM configuration.
        """
        self.db_dir = db_dir
        
        # 1. Initialize Gemini API key
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. Please set it before initializing."
            )
        genai.configure(api_key=api_key)
        self.llm = genai.GenerativeModel(llm_model)

        # 2. Setup Embeddings (Pin model, CPU usage, normalized outputs)
        print(f"Loading embedding model: {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        # 3. Setup Text Splitter (Balanced chunk sizes)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            keep_separator=True
        )

        # 4. Setup Reranker (Cross-Encoder)
        print(f"Loading reranker model: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model, device="cpu")

        # 5. Initialize/Load Vector Store (Chroma)
        print(f"Connecting to Chroma DB at: {self.db_dir}...")
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory=self.db_dir
        )
        print("[OK] RAG Chatbot initialized successfully.")

    def ingest_pdf(self, pdf_path):
        """
        Parses a PDF file, splits it into chunks, calculates embeddings, and updates the vector database.
        Includes duplicate prevention: if the file was already ingested, its old chunks are replaced.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        filename = os.path.basename(pdf_path)
        print(f"\n[Ingest] Extracting text from PDF: {pdf_path}...")
        
        reader = pypdf.PdfReader(pdf_path)
        documents = []
        ids = []

        # Calculate the size of the PDF in characters to update the chunking parameters
        total_chars = 0
        for page in reader.pages:
            text = page.extract_text()
            if text:
                total_chars += len(text)
        
        print(f"[Ingest] Calculated PDF text size: {total_chars} characters.")

        # Update text splitter to chunk by the size of the PDF
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(1, total_chars),
            chunk_overlap=0,
            length_function=len,
            keep_separator=True
        )

        # Extract text page-by-page
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text or not text.strip():
                continue

            # Split the page text into chunks
            chunks = self.text_splitter.split_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "page": page_idx + 1
                    }
                )
                documents.append(doc)
                # Deterministic IDs to help avoid duplicate storage
                ids.append(f"{filename}_p{page_idx + 1}_c{chunk_idx}")

        if not documents:
            print("[Ingest] Warning: No text could be extracted from this document.")
            return

        print(f"[Ingest] Created {len(documents)} chunks from {filename}.")

        # Prevent duplicates: delete previous entries of the same source before re-adding
        existing = self.vector_store.get(where={"source": filename})
        if existing and existing.get("ids"):
            old_ids = existing["ids"]
            self.vector_store.delete(ids=old_ids)
            print(f"[Ingest] Removed {len(old_ids)} existing chunks for {filename} to prevent duplicates.")

        # Save to database
        self.vector_store.add_documents(documents=documents, ids=ids)
        print(f"[Ingest] Successfully ingested & indexed '{filename}'.")

    def retrieve_and_rerank(self, query, initial_k=15, final_k=3, filter_dict=None):
        """
        Performs a two-stage retrieval:
        1. Retrieval: Embedding-based similarity search to find top candidates.
        2. Reranking: Cross-Encoder model scores candidate-query pairs, reordering the top candidates.
        
        Returns:
            list: A list of dicts containing the top final_k documents and metadata:
                  [{'document': Document, 'score': float}, ...]
        """
        # Stage 1: Initial retrieval
        print(f"\n[Search] Performing initial semantic search (k={initial_k})...")
        search_kwargs = {"k": initial_k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
            
        candidates = self.vector_store.similarity_search(query, **search_kwargs)
        if not candidates:
            print("[Search] No candidate documents retrieved.")
            return []

        # Stage 2: Reranking using Cross-Encoder
        print(f"[Search] Reranking {len(candidates)} candidates using Cross-Encoder...")
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # Pair candidates with their rerank scores
        candidate_scores = list(zip(candidates, scores))
        # Sort descending based on reranking score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # Slice to final_k
        top_results = candidate_scores[:final_k]
        
        # Format results
        formatted_results = []
        for doc, score in top_results:
            formatted_results.append({
                "document": doc,
                "score": float(score)
            })
            
        return formatted_results

    def generate_answer(self, question, context_results):
        """
        Constructs the grounded context, builds the system prompt, and calls Gemini LLM.
        """
        if not context_results:
            return "The vector database does not contain any relevant context to answer this question."

        # Construct structured context text
        context_blocks = []
        for item in context_results:
            doc = item["document"]
            score = item["score"]
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            context_blocks.append(
                f"[Source: {source} (Page {page}), Relevance Score: {score:.4f}]\nContent: {doc.page_content}"
            )
        
        context_str = "\n\n".join(context_blocks)

        # Formulate grounded instructions
        prompt = f"""You are a reliable AI Chatbot. Your answers must be based strictly on the provided Context.

Context:
{context_str}

Question:
{question}

Instructions:
1. Ground your answer ONLY on the provided Context.
2. If the answer cannot be found in the Context, respond exactly with: "I cannot find the answer in the provided documents."
3. Do NOT make up, extrapolate, or assume any information outside the Context.
4. Keep your answer clear, factual, and concise. Add page citations where applicable.
"""
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred while generating the answer: {e}"

    def ask(self, question, initial_k=15, final_k=3, filter_dict=None):
        """
        Unified endpoint linking retrieval, reranking, and generation.
        """
        # Retrieve & Rerank
        context_results = self.retrieve_and_rerank(
            query=question,
            initial_k=initial_k,
            final_k=final_k,
            filter_dict=filter_dict
        )
        
        # Generate grounded response
        answer = self.generate_answer(question, context_results)
        
        return {
            "answer": answer,
            "sources": context_results
        }


# --- Demonstration Runner ---
if __name__ == "__main__":
    # Example execution testing the chatbot class
    print("=== Starting RAG Chatbot Runner ===")
    
    # Check if API key is in environment, fallback to secrets.toml if missing
    if "GEMINI_API_KEY" not in os.environ:
        try:
            import tomllib
            SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
            secrets_path = os.path.join(SCRIPT_DIR, ".streamlit", "secrets.toml")
            if os.path.exists(secrets_path):
                with open(secrets_path, "rb") as f:
                    secrets = tomllib.load(f)
                    if "GEMINI_API_KEY" in secrets:
                        os.environ["GEMINI_API_KEY"] = secrets["GEMINI_API_KEY"]
        except Exception:
            pass

    if "GEMINI_API_KEY" not in os.environ:
        print("WARNING: GEMINI_API_KEY environment variable not found.")
        print("Please export/set GEMINI_API_KEY or configure secrets.toml before running this script.")
        exit(1)

    # Initialize chatbot
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    chatbot = ClassBasedRAGChatbot(
        db_dir=os.path.join(SCRIPT_DIR, "test_rag_db"),
        llm_model="gemini-2.5-flash"
    )

    # Test PDF file (Change this path if needed)
    test_pdf_path = os.path.join(SCRIPT_DIR, "webdev-tutorial.pdf")
    
    if os.path.exists(test_pdf_path):
        # Ingest PDF
        chatbot.ingest_pdf(test_pdf_path)
        
        # Test Query
        test_query = "What is WEBDEV and how does the tutorial guide users?"
        print(f"\n[Test Query] Asking: '{test_query}'")
        
        result = chatbot.ask(test_query, initial_k=5, final_k=2)
        
        print("\n=== Bot Answer ===")
        print(result["answer"])
        print("\n=== References used ===")
        for i, item in enumerate(result["sources"], 1):
            doc = item["document"]
            print(f"{i}. [Page {doc.metadata.get('page')}] (Score: {item['score']:.4f}): {doc.page_content[:150]}...")
    else:
        print(f"\nNote: Test PDF not found at {test_pdf_path}. Skipping ingestion demo.")
        print("You can verify by calling chatbot.ingest_pdf('path/to/your/pdf.pdf') manually.")
