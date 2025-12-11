from search import search_prompt

def main():
    print("Iniciando chat...")
    print("Digite 'sair' para encerrar.\n")

    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    while True:
        try:
            question = input("PERGUNTA: ").strip()

            if not question:
                continue

            if question.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando chat...")
                break

            response = chain.invoke(question)
            print(f"RESPOSTA: {response}\n")
            print("---\n")

        except KeyboardInterrupt:
            print("\n\nEncerrando chat...")
            break
        except Exception as e:
            print(f"Erro ao processar pergunta: {str(e)}\n")

if __name__ == "__main__":
    main()