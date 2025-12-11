import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings():
    if OPENAI_API_KEY:
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
    elif GOOGLE_API_KEY:
        return GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
    else:
        raise ValueError(
            "Nenhuma API key configurada. Configure OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env"
        )


def get_llm():
    if OPENAI_API_KEY:
        return ChatOpenAI(
            model="gpt-5-nano",
            api_key=OPENAI_API_KEY,
            temperature=0
        )
    elif GOOGLE_API_KEY:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=GOOGLE_API_KEY,
            temperature=0
        )
    else:
        raise ValueError(
            "Nenhuma API key configurada. Configure OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env"
        )


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def search_prompt(question=None):
    try:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL não configurado no arquivo .env")
        if not COLLECTION_NAME:
            raise ValueError("PG_VECTOR_COLLECTION_NAME não configurado no arquivo .env")

        embeddings = get_embeddings()
        llm = get_llm()

        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        chain = (
            {"contexto": retriever | format_docs, "pergunta": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    except Exception as e:
        print(f"Erro ao inicializar search_prompt: {str(e)}")
        return None