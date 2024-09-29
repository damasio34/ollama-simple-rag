import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain

#https://dev.to/mohsin_rashid_13537f11a91/rag-with-ollama-1049

model = "llama3.1"
base_url="http://127.0.0.1:11434"

# def souce_loading:
#     pass

# def text_splitting:
#     pass

# def embbeding_storage:
#     pass

# def retriever:
#     pass    

# def output:
#     pass

# Loading The LLM (Language Model)
llm = Ollama(
    model = model, 
    base_url = base_url,
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
)

# Setting Ollama Embeddings
embed_model = OllamaEmbeddings(
    model = model,
    base_url = base_url
)

# Loading Text
with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read().replace('\n', '')

# Splitting Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 128
)
chunks = text_splitter.split_text(text)

# Creating a Vector Store (Chroma) from Text
vector_store = Chroma.from_texts(chunks, embed_model)

# Creating a Retriever
retriever = vector_store.as_retriever()

# Creating a Retrieval Chain
chain = create_retrieval_chain(retriever = retriever, combine_docs_chain = llm)

# print(response['answer'])

while True:
    query = input("\nPergunta: ")
    if query == "sair":
        break
    if query.strip() == "":
        continue

    # Prompt
    template = """
    Responda sempre de forma objetiva, com um pouco de informação.
    Caso não saiba a responsta, responda: 'Não sei'.
    {context}
    Pergunta: {question}
    Resposta: """

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # llm = Ollama(model="llama3.1", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    # Retrieval-QA Chat Prompt
    retrieval_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type_kwargs = {"prompt": QA_CHAIN_PROMPT}
    )
    # retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # Combining Documents
    # combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    # Final Retrieval Chain
    # retrieval_chain = create_retrieval_chain(retriever = retriever, combine_docs_chain = combine_docs_chain)

    # # Invoking the Retrieval Chain
    # response = retrieval_chain.invoke({
    #     "input": "Quem é Peludo?"
    # })

    # result = qa_chain.invoke({"query": query})

    result = retrieval_chain.invoke({"query": query})
    print(result['result'])