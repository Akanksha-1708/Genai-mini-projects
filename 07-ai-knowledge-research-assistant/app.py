from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

files = [
    "documents/genai_notes.txt",
    "documents/langchain_notes.txt",
    "documents/project_notes.txt"
]

documents = []

for file_path in files:
    loader = TextLoader(
        file_path,
        encoding="utf-8"
    )
    documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print("=" * 70)
print("KNOWLEDGE BASE PROCESSING")
print("=" * 70)
print(f"Documents loaded : {len(documents)}")
print(f"Chunks created   : {len(chunks)}")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="knowledge_base"
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)

print("\nKnowledge base created successfully!")
print("Retriever is ready!")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai",
    max_new_tokens=500,
    temperature=0.3
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful AI knowledge assistant.
Answer the user's question using ONLY the provided context.
If the answer is not available in the context, say:
"I don't have enough information in the knowledge base."
Do not invent information.
Context:
{context}
Question:
{question}
Answer:
"""
)

parser = StrOutputParser()
chain = prompt | model | parser

def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )

def answer_question(question):
    relevant_documents = retriever.invoke(question)
    context = format_documents(relevant_documents)
    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )
    return answer

search_tool=TavilySearch(max_results=5)

def web_search(query):
    result=search_tool.invoke({"query":query})
    return result

def main():
    print("\n" + "=" * 70)
    print("AI KNOWLEDGE & RESEARCH ASSISTANT")
    print("=" * 70)
    print("\nAsk questions about the knowledge base.")
    print("Use 'search:' for web search.")
    print("Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "exit":
            print("\nGoodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue
        if question.lower().startswith("search:"):
            query=question[7:].strip()
            if not query:
                print("Please enter a search query.")
                continue
            try:
                print("\nSearching the web...")
                result=web_search(query)
                print("\nSearch Results:")
                print(result)
            except Exception as error:
                print(f"\nSearch Error:{error}")
            continue

        try:
            print("\nSearching knowledge base...")
            answer = answer_question(question)
            print("\nAssistant:")
            print(answer)
        except Exception as error:
            print(f"\nError: {error}")

if __name__ == "__main__":
    main()