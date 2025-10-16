# #!/usr/bin/env python3
# from rag_chain import create_rag_chain
# import sys


# def main():
#     print("free langchain rag chatbot")
#     print("="*40)

#     try:
#         rag_chain = create_rag_chain()
#     except FileNotFoundError as e:
#         print(f"Error:{e}")
#         return
    
#     print("\n ask questions about your documents")
#     print("type 'quit' or 'exit' to stop\n")

#     while True:
#         try:
#             question = input("Question:").strip()

#             if question.lower() in ['quit','exit','q']:
#                 print("goodbye")
#                 break
            
#             if not question:
#                 continue
            
#             result = rag_chain.query(question)



#             print(f'answer')
#             print(result["answer"])


#             if result["sources"]:
#                 print(f"\nsources:")
#                 for sources in result["sources"]:
#                     print(f"{sources}")

#             print("\n"+"-"*50+"\n")

#         except KeyboardInterrupt:
#             print("goodbye")
#             break
#         except Exception as e:
#             print(f"error :{e}")
#             continue


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import sys
from rag_chain import create_rag_chain


def main():
    print("\nFree LangChain RAG Chatbot")
    print("=" * 45)

    # Try initializing the RAG chain safely
    try:
        rag_chain = create_rag_chain()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Tip: Run 'ingest.py' to build your Chroma database first.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nFailed to initialize RAG chain: {e}\n")
        sys.exit(1)

    print("\nAsk questions about your documents.")
    print("Type 'quit' or 'exit' to stop.\n")

    # Main interaction loop
    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye, explorer.")
                break

            if not question:
                continue

            result = rag_chain.query(question)

            print("\nAnswer:")
            print(result.get("answer", "No answer found."))

            sources = result.get("sources", [])
            if sources:
                print("\nSources:")
                for src in sources:
                    print(f"  - {src}")
            else:
                print("\n(No source documents found.)")

            print("\n" + "-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted — exiting cleanly.")
            break
        except EOFError:
            print("\nEOF received. Goodbye.")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Continuing...\n")
            continue


if __name__ == "__main__":
    main()
