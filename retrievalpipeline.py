import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# Configuration
PERSISTENT_DIRECTORY = "db/chroma_db"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

def main():
    # 1. Initialize the embedding model
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    # 2. Load the existing vector store
    if not os.path.exists(PERSISTENT_DIRECTORY):
        print(f"Error: Directory {PERSISTENT_DIRECTORY} does not exist.")
        return

    db = Chroma(
        persist_directory=PERSISTENT_DIRECTORY,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    # 3. Define your list of synthetic questions
    queries = [
        "What was NVIDIA's first graphics accelerator called?",
        "Which company did NVIDIA acquire to enter the mobile processor market?",
        "What was Microsoft's first hardware product release?",
        "How much did Microsoft pay to acquire GitHub?",
        "In what year did Tesla begin production of the Roadster?",
        "Who succeeded Ze'ev Drori as CEO in October 2008?",
        "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?",
        "What was the original name of Microsoft before it became Microsoft?"
    ]

    # 4. Set up the retriever
    # Using 'k=3' for brevity in output, but you can change this back to 5
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 5. Execute queries
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*20} Query {i} {'='*20}")
        print(f"Question: {query}")
        
        relevant_docs = retriever.invoke(query)
        
        print("\n--- Top Relevant Context ---")
        if not relevant_docs:
            print("No relevant documents found.")
        else:
            for j, doc in enumerate(relevant_docs, 1):
                # Clean up whitespace for cleaner terminal output
                content = doc.page_content.replace('\n', ' ').strip()
                print(f"[{j}] {content[:200]}...") 

if __name__ == "__main__":
    main()