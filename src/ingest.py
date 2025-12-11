import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
GOOGLE_EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL")


def get_embeddings():
    if OPENAI_API_KEY:
        print(f"Usando OpenAI Embeddings: {OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
    elif GOOGLE_API_KEY:
        print(f"Usando Google Embeddings: {GOOGLE_EMBEDDING_MODEL}")
        return GoogleGenerativeAIEmbeddings(
            model=GOOGLE_EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
    else:
        raise ValueError(
            "Nenhuma API key configurada. Configure OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env"
        )


def ingest_pdf():
    try:
        if not PDF_PATH:
            raise ValueError("PDF_PATH não configurado no arquivo .env")
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL não configurado no arquivo .env")
        if not COLLECTION_NAME:
            raise ValueError("PG_VECTOR_COLLECTION_NAME não configurado no arquivo .env")

        print(f"Iniciando ingestão do PDF: {PDF_PATH}")

        print("Carregando PDF...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"PDF carregado: {len(documents)} página(s)")

        print("Dividindo documento em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Documento dividido em {len(chunks)} chunks")

        embeddings = get_embeddings()

        print("Armazenando chunks no banco de dados...")
        vectorstore = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DATABASE_URL,
            pre_delete_collection=True,
        )

        print(f"Ingestão concluída com sucesso!")
        print(f"{len(chunks)} chunks armazenados")
        print(f"Collection: {COLLECTION_NAME}")

    except FileNotFoundError:
        print(f"Erro: Arquivo PDF não encontrado: {PDF_PATH}")
    except Exception as e:
        print(f"Erro durante a ingestão: {str(e)}")
        raise


if __name__ == "__main__":
    ingest_pdf()