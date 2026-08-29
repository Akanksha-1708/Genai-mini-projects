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
from langchain_core.tools import tool
from langchain.agents import create_agent

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

@tool
def knowledge_base_search(question: str) -> str:
    """
    Search the personal knowledge base for information.
    Use this tool when the user asks about information
    that may be contained in the stored documents,
    including GenAI concepts, LangChain concepts,
    and projects in the GenAI Learning Journey.
    """
    relevant_documents = retriever.invoke(question)
    if not relevant_documents:
        return "No relevant information found in the knowledge base."
    context = "\n\n".join(
        document.page_content
        for document in relevant_documents
    )
    return context

search_tool = TavilySearch(
    max_results=5
)
@tool
def web_search(query: str) -> str:
    """
    Search the web for current or external information.
    Use this tool when the user asks for recent,
    current, real-world, or information that is
    unlikely to be available in the knowledge base.
    """
    result = search_tool.invoke(
        {
            "query": query
        }
    )
    return str(result)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai",
    max_new_tokens=500,
    temperature=0.3
)
model = ChatHuggingFace(
    llm=llm
)

tools = [
    knowledge_base_search,
    web_search
]

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="""
You are an AI Knowledge and Research Assistant.
You have access to two tools:
1. knowledge_base_search
   - Searches the user's personal knowledge base.
   - Use it for questions about the stored documents,
     GenAI concepts, LangChain concepts, and projects.
2. web_search
   - Searches the internet.
   - Use it for current, recent, external, or
     information not available in the knowledge base.
Choose the appropriate tool based on the user's question.
You may use more than one tool when necessary.
For example:
- If the user asks "What is RAG?",
  use the knowledge base.
- If the user asks "What are the latest developments
  in Generative AI?",
  use web search.
- If the user asks a question requiring both stored
  information and current information, use both tools.
Do not invent information.
Use the information returned by the tools to produce
the final answer.
"""
)

def main():
    print("\n" + "=" * 70)
    print("AI KNOWLEDGE & RESEARCH ASSISTANT")
    print("=" * 70)
    print("\nThe assistant can:")
    print("- Search your knowledge base")
    print("- Search the web")
    print("- Use multiple tools when necessary")
    print("\nType 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "exit":
            print("\nGoodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue
        try:
            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }
            )
            print("\nAgent messages:")
            for message in response["messages"]:
                message_type = type(message).__name__
                if message_type == "HumanMessage":
                    print(
                        "HumanMessage ->",
                        message.content
                    )
                elif message_type == "AIMessage":
                    if message.tool_calls:
                        print(
                            "AIMessage -> Tool calls:",
                            [
                                call["name"]
                                for call in message.tool_calls
                            ]
                        )
                    elif message.content:
                        print(
                            "AIMessage ->",
                            message.content
                        )
                elif message_type == "ToolMessage":
                    print(
                        "ToolMessage ->",
                        message.content
                    )
            print("\nAssistant:")
            print(
                response["messages"][-1].content
            )
        except Exception as error:
            print(f"\nError: {error}")

if __name__ == "__main__":
    main()