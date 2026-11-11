# 🚀 Generative AI Mini Projects

A collection of hands-on Generative AI projects built while learning **Python, LangChain, Large Language Models (LLMs), embeddings, Retrieval-Augmented Generation (RAG), tool calling, and AI agents**.

This repository follows a gradual learning path: starting with prompt-based applications and moving towards retrieval-based systems and autonomous multi-tool agents.

Each project focuses on a specific GenAI concept and demonstrates how it can be used to build practical AI applications.

---

## 📚 Projects

| # | Project | Main Concepts |
|---|---|---|
| 01 | [AI Profile Generator](./01-ai-profile-generator/) | Prompt Templates, LCEL, RunnableParallel, Pydantic, Output Parsers |
| 02 | [AI PDF Summarizer](./02-ai-pdf-summarizer/) | PDF Loading, Text Splitting, Prompt Templates, Hugging Face |
| 03 | [AI Semantic Search](./03-ai-semantic-search/) | Embeddings, Sentence Transformers, Chroma, Similarity Search |
| 04 | [Personal Knowledge RAG](./04-personal-knowledge-rag/) | Document Loaders, Chunking, Vector Stores, Retrievers, RAG |
| 05 | [Tool-Calling AI Assistant](./05-tool-calling-assistant/) | Custom Tools, `@tool`, `bind_tools()`, Tool Calls, Tool Messages |
| 06 | [Multi-Tool AI Agent](./06-multi-tool-ai-agent/) | `create_agent()`, Tool Selection, Agent Loop, Multi-Step Execution |
| 07 | [AI Knowledge & Research Assistant](./07-ai-knowledge-research-assistant/) | RAG, Chroma, Tavily Search, Agents, Multiple Tools |

---

## 🧠 Project Details

### 01. AI Profile Generator

**Goal:** Generate multiple professional profile formats from a single set of user details.

The application collects information such as name, education, skills, projects, experience, achievements, career goals and interests.

It uses separate prompt templates to generate six types of professional content:

- LinkedIn profile summary
- Resume summary
- GitHub profile README
- Portfolio About section
- Professional biography
- Elevator pitch

**How it works:**

1. Collects user information through the terminal.
2. Validates the input using a Pydantic model.
3. Passes the information to six different prompt chains.
4. Uses `RunnableParallel` to execute the chains together.
5. Validates the generated output using another Pydantic schema.
6. Saves the results as JSON and Markdown files.

**Technologies:** Python, LangChain, LCEL, Hugging Face, Pydantic, RunnableParallel.

### 02. AI PDF Summarizer

**Goal:** Convert a PDF document into an easy-to-understand structured summary.

The application loads a PDF using LangChain's `PyPDFLoader` and divides its content into smaller chunks using `RecursiveCharacterTextSplitter`.

It then sends the combined extracted text to the Qwen 2.5 7B Instruct model through Hugging Face.

The generated summary contains:

- Executive summary
- Key points
- Important concepts
- Keywords
- Final takeaways

The output is displayed in the terminal and saved in both `.txt` and `.md` formats.

**Technologies:** Python, LangChain, PyPDF, Hugging Face, Qwen 2.5 7B, PromptTemplate, RecursiveCharacterTextSplitter, StrOutputParser.

### 03. AI Semantic Search

**Goal:** Search documents based on meaning rather than exact keyword matching.

This project demonstrates how text can be converted into vector embeddings and searched using semantic similarity.

**How it works:**

1. Loads text from `sample.txt`.
2. Splits the document into chunks.
3. Converts chunks into embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. Stores the embedded documents in a Chroma vector store.
5. Accepts a natural-language query.
6. Retrieves the three most similar document chunks.

**Technologies:** Python, LangChain, Hugging Face Sentence Transformers, Chroma, Embeddings, Similarity Search.

### 04. Personal Knowledge RAG

**Goal:** Build a question-answering assistant that responds using information from a personal collection of documents.

The application loads three knowledge files:

- `ai_notes.txt`
- `langchain_notes.txt`
- `projects.txt`

It processes these documents, creates embeddings, stores them in Chroma and retrieves relevant chunks whenever a question is asked.

The retrieved content is passed to Qwen 2.5 7B along with the user's question.

The prompt instructs the model to answer only from the supplied context and to say when the information is insufficient.

**How it works:**

`Documents → Text Splitting → Embeddings → Chroma → Retriever → Prompt + LLM → Answer`

The application supports multiple questions in one terminal session until the user enters `exit`.

**Technologies:** Python, LangChain, RAG, Hugging Face, Qwen 2.5 7B, Chroma, Sentence Transformers, Retrievers.

### 05. Tool-Calling AI Assistant

**Goal:** Allow an LLM to select and execute tools based on a user's request.

This project introduces function/tool calling, where the model decides whether a tool is needed and supplies its arguments.

Three custom tools are implemented:

| Tool | Purpose |
|---|---|
| Calculator | Evaluates mathematical expressions |
| Currency Converter | Converts between supported currencies using sample fixed rates |
| Weather | Returns sample weather information for selected cities |

**How it works:**

1. Defines tools using LangChain's `@tool` decorator.
2. Registers the tools with the model using `bind_tools()`.
3. Sends the user's question to the model.
4. Checks whether the model requested any tool calls.
5. Executes the selected tools with their supplied arguments.
6. Sends the tool results back to the model.
7. Generates the final response.

**Technologies:** Python, LangChain, Hugging Face, Qwen 2.5 7B, Custom Tools, Tool Calling, HumanMessage, ToolMessage.

### 06. Multi-Tool AI Agent

**Goal:** Move beyond individual tool calls and build an agent capable of selecting tools and completing multi-step tasks.

This project uses LangChain's `create_agent()` to create an agent with access to multiple tools.

The available tools are:

- Calculator
- Currency converter
- Sample weather lookup

Unlike the previous project, the agent manages the tool-selection and execution loop.

For example, a user can ask the agent to convert an amount into another currency and then add a particular value to the converted amount.

The agent can:

1. Understand the user's request.
2. Select the appropriate tool.
3. Execute the tool.
4. Read and interpret the result.
5. Pass the result into another tool when required.
6. Return the final answer.

**Technologies:** Python, LangChain Agents, `create_agent()`, Hugging Face, Qwen 2.5 7B, Custom Tools.

### 07. AI Knowledge & Research Assistant

**Goal:** Build an AI assistant that can retrieve information from personal documents and search the web for external information.

This project combines two tools:

**1. Knowledge Base Search**
- Loads GenAI, LangChain and project notes.
- Splits documents into chunks.
- Generates embeddings.
- Stores the documents in Chroma.
- Retrieves relevant content using a retriever.

**2. Web Search**
- Uses Tavily Search to retrieve external information.
- Helps answer questions involving current or external topics.

Both tools are registered with a LangChain agent using `create_agent()`.

The agent decides which tool to use based on the question. It can also use both tools when a question requires information from the personal knowledge base as well as the web.

**Example behaviour:**

| User Question | Expected Tool |
|---|---|
| What is RAG? | Knowledge Base Search |
| Explain my stored LangChain notes. | Knowledge Base Search |
| What are the latest developments in GenAI? | Web Search |
| Compare my stored notes with recent developments. | Both tools |

**Technologies:** Python, LangChain, LangChain Agents, RAG, Chroma, Hugging Face, Sentence Transformers, Tavily Search.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| LLM Framework | LangChain |
| Model Provider | Hugging Face |
| Language Model | Qwen 2.5 7B Instruct |
| Embedding Model | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Database | Chroma |
| Data Validation | Pydantic |
| Document Processing | PyPDF, LangChain Document Loaders |
| Text Processing | RecursiveCharacterTextSplitter |
| Web Search | Tavily |
| Environment Management | Python Dotenv |

---

## 📂 Repository Structure

```text
Genai-mini-projects/
│
├── 01-ai-profile-generator/
│   ├── chains/
│   │   └── profile_generator.py
│   ├── models/
│   │   ├── input_schema.py
│   │   └── output_schema.py
│   ├── prompts/
│   │   └── prompt.py
│   ├── app.py
│   └── requirements.txt
│
├── 02-ai-pdf-summarizer/
│   ├── sample_pdfs/
│   │   └── sample.pdf
│   ├── outputs/
│   ├── app.py
│   ├── pdf_loader.py
│   └── requirements.txt
│
├── 03-ai-semantic-search/
│   ├── app.py
│   ├── sample.txt
│   └── requirements.txt
│
├── 04-personal-knowledge-rag/
│   ├── documents/
│   │   ├── ai_notes.txt
│   │   ├── langchain_notes.txt
│   │   └── projects.txt
│   ├── app.py
│   └── requirements.txt
│
├── 05-tool-calling-assistant/
│   ├── app.py
│   └── requirements.txt
│
├── 06-multi-tool-ai-agent/
│   ├── app.py
│   └── requirements.txt
│
├── 07-ai-knowledge-research-assistant/
│   ├── documents/
│   │   ├── genai_notes.txt
│   │   ├── langchain_notes.txt
│   │   └── project_notes.txt
│   ├── app.py
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Akanksha-1708/Genai-mini-projects.git

cd Genai-mini-projects
```

### 2. Create a virtual environment

Each project can be run independently. Create a separate virtual environment for the project you want to execute.

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

Navigate into the project folder and install its dependencies.

For projects with valid requirements files:

```bash
pip install -r requirements.txt
```

**Note:** Project 01 currently has a requirements file written as `pip install` commands rather than package requirement entries. Until that file is corrected, install its dependencies manually:

```bash
pip install langchain langchain-core python-dotenv pydantic langchain-huggingface huggingface_hub
```

Project 03 currently has an empty requirements file. Install its dependencies manually:

```bash
pip install langchain-core langchain-text-splitters langchain-huggingface langchain-chroma sentence-transformers
```

### 4. Configure environment variables

Projects using Hugging Face models require an appropriate Hugging Face access token.

Create a `.env` file in the relevant project directory:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
```

For Project 07, also configure Tavily:

```env
TAVILY_API_KEY=your_tavily_api_key
```

Keep API keys private. Do not commit `.env` files to GitHub.

### 5. Run a project

Navigate into the relevant project directory before running its application. This is important because the scripts use relative paths for their input documents and output files.

For example:

```bash
cd 03-ai-semantic-search
python app.py
```

For another project, replace the folder name accordingly.

---

## 🔑 Important Notes & Current Scope

- The projects are independent learning applications, not one combined application.
- Projects 02, 04, 05, 06 and 07 use Hugging Face models and may require valid API access.
- Project 05 uses fixed sample currency rates and predefined weather data; these are demonstrations, not live APIs.
- Project 02 currently combines the extracted chunks into one model request. Very large PDFs may exceed model input limits.
- Projects 03, 04 and 07 currently create Chroma stores in memory rather than persisting them between executions.
- Project 07 requires a working Tavily API key for web search.
- Model availability, provider configuration, API quotas and dependency versions can affect execution.

---

## 🎯 Learning Outcomes

Through these projects, I have explored and practised:

- Building LLM applications using LangChain.
- Designing reusable prompt templates.
- Understanding LCEL and parallel execution with `RunnableParallel`.
- Validating structured input and output using Pydantic.
- Loading and processing PDF and text documents.
- Creating embeddings and performing semantic search.
- Understanding vector databases and similarity retrieval.
- Implementing Retrieval-Augmented Generation (RAG).
- Creating custom tools and connecting them to LLMs.
- Understanding the difference between tool calling and agent-based execution.
- Building multi-tool agents that can perform multi-step tasks.
- Combining private knowledge retrieval with external web search.

---

## 👩‍💻 Author

**Akanksha Dwivedi**

B.Tech Computer Science Engineering Student | Exploring Generative AI, Machine Learning and Agentic AI.

- GitHub: [Akanksha-1708](https://github.com/Akanksha-1708)
- Portfolio: [My Portfolio](https://github.com/Akanksha-1708/Portfolio)

---

⭐ If you find this repository useful, feel free to explore the individual projects and follow the learning journey.
