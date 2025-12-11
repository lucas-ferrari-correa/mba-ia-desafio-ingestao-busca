# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de ingestão e busca semântica com LangChain e PostgreSQL + pgVector.

## Descrição

Este projeto implementa um sistema de RAG (Retrieval-Augmented Generation) que permite:
- Ingerir documentos PDF em um banco de dados vetorial
- Realizar buscas semânticas baseadas no conteúdo do PDF
- Interagir via CLI para fazer perguntas e receber respostas contextualizadas

## Tecnologias

- Python 3.x
- LangChain
- PostgreSQL + pgVector
- Docker & Docker Compose
- OpenAI API ou Google Gemini API

## Pré-requisitos

- Docker e Docker Compose instalados
- Python 3.x instalado
- Chave de API da OpenAI ou Google Gemini

## Instalação

### 1. Clone o repositório

```bash
git clone <seu-repositorio>
cd mba-ia-desafio-ingestao-busca
```

### 2. Crie um ambiente virtual Python

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Edite o arquivo `.env` e adicione sua chave de API:

**Para OpenAI:**
```
OPENAI_API_KEY=sk-...
```

**Para Google Gemini:**
```
GOOGLE_API_KEY=...
```

As demais variáveis já estão configuradas com valores padrão.

## Execução

### 1. Subir o banco de dados PostgreSQL

```bash
docker compose up -d
```

Aguarde alguns segundos para o banco inicializar completamente.

### 2. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Este comando irá:
- Carregar o arquivo `document.pdf`
- Dividir o conteúdo em chunks de 1000 caracteres com overlap de 150
- Gerar embeddings
- Armazenar no banco de dados PostgreSQL com pgVector

### 3. Executar o chat interativo

```bash
python src/chat.py
```

Agora você pode fazer perguntas sobre o conteúdo do PDF.

## Exemplo de uso

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Estrutura do projeto

```
.
├── docker-compose.yml          # Configuração do PostgreSQL + pgVector
├── requirements.txt            # Dependências Python
├── .env.example               # Template de variáveis de ambiente
├── .env                       # Variáveis de ambiente (criar)
├── src/
│   ├── ingest.py             # Script de ingestão do PDF
│   ├── search.py             # Lógica de busca semântica
│   └── chat.py               # Interface CLI
├── document.pdf              # PDF para ingestão
└── README.md                 # Este arquivo
```

## Parar o banco de dados

```bash
docker compose down
```

Para remover os dados persistidos:

```bash
docker compose down -v
```