import os
import time
from typing import List

import streamlit as st
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.llms.exllamav2 import ExLlamaV2
from langchain_community.llms.vllm import VLLMOpenAI, VLLM
from langchain_community.llms import OpenAI
# from exllamav2.generator import ExLlamaV2Sampler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from utils import print_messages, StreamHandler

# API KEY 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        time.sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        time.sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


st.set_page_config(page_title="Crowdworks LLM Demo Streamlit", page_icon="🦜")
session_id = "1"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 채팅 대화기록을 저장하는 store 세션 상태 변수
if"store" not in st.session_state:
    st.session_state["store"] = {}


with st.sidebar:
    llm_option = st.selectbox("Choose LLM",
                              (
                                  "NurtureAI/Meta-Llama-3-8B-Instruct-32k",
                                  "ChatGPT",
                                  "Llama-3-70B-Instruct-32k-cw_data_20240503-Q4_K_M-GGUF",
                                  "Llama-3-70B-Instruct-32k-cw_data_20240508-Q3_K-GGUF",
                                  "Llama-3-70B-Instruct-32k-cw_data_2040508-4.0bpw-exl2",
                              ))
    embedding_option = st.selectbox("Choose Embedding",
                                    (
                                        "BM-K/KoSimCSE-roberta",
                                        "text-embedding-3-small",
                                    ))

    docs_option = st.selectbox("Choose Docs",
                               (
                                   None,
                                   "web_page",
                                   "pdf",  # TODO PDF 로더 추가하기
                               ))
    clear_btn = st.button("대화기록 초기화")

    if docs_option == 'web_page':
        if url := st.text_input("url 을 입력해주세요."):
            loader = WebBaseLoader(url)
            docs = loader.load()
            st.session_state["store"]["docs"] = docs
        if collection_name := st.text_input("Collection Name"):
            st.session_state["store"]["collection_name"] = collection_name

    if clear_btn:
        st.session_state["messages"] = []
        if st.session_state["store"].get(session_id):
            del st.session_state["store"][session_id]
        if st.session_state["store"].get("docs"):
            del st.session_state["store"]["docs"]
        if st.session_state["store"].get("retriever"):
            del st.session_state["store"]['retriever']
        if st.session_state["store"].get("collection_name"):
            del st.session_state["store"]["collection_name"]
        st.experimental_rerun()


st.title(f"🦜 {llm_option}")

# 이전 대화기록을 출력해 주는 코드
print_messages()


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]  # 해당 세션 ID에 대한 세션 기록 반환


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_question(input):
    if not input:
        return None
    elif isinstance(input, str):
        return input
    elif isinstance(input, dict) and 'question' in input:
        return input['question']


if user_input := st.chat_input("메시지를 입력해 주세요."):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    # st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    # AI의 답변
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())

        # LLM을 사용하여 AI의 답변을 생성
        # 1. 모델 생성
        if llm_option == "ChatGPT":
            llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
        elif llm_option == "Llama-3-70B-Instruct-32k-cw_data_20240503-Q4_K_M-GGUF":
            llm = ChatOllama(
                model="llama-3-70B-Instruct-32k-cw_data_20240503:latest",
                streaming=True,
                callbacks=[stream_handler],
            )
        elif llm_option == "Llama-3-70B-Instruct-32k-cw_data_20240508-Q3_K-GGUF":
            llm = ChatOllama(
                model="llama-3-70B-Instruct-32k-cw_data_20240508:latest",
                streaming=True,
                callbacks=[stream_handler],
            )
        # elif llm_option == "Llama-3-70B-Instruct-32k-cw_data_2040508-4.0bpw-exl2":
        #     settings = ExLlamaV2Sampler.Settings(temperature=0.7,
        #                                          top_k=-1,
        #                                          top_p=0.8,
        #                                          token_repetition_penalty=1.05)
        #     llm = ExLlamaV2(
        #         model_path="../model_tuning/quantized_model/20240509/Llama-3-70B-Instruct-32k-cw_data_2040508-exl2/4.0bpw",
        #         streaming=True,
        #         callbacks=[stream_handler],
        #         max_new_tokens=124,
        #         settings=settings,
        #         stop_sequences=["<|eot_id|>", "<|begin_of_text|>"]
        #     )
        elif llm_option == "NurtureAI/Meta-Llama-3-8B-Instruct-32k":
            # 따로 서버에서 vllm 모델을 실행해야 함
            llm = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                model_name="NurtureAI/Meta-Llama-3-8B-Instruct-32k",
                callbacks=[stream_handler],
                streaming=True,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                temperature=0.7,
                top_p=0.8,
                max_tokens=1024,
            )

        if st.session_state["store"].get('retriever') is None and st.session_state["store"].get("docs"):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0,
                length_function=len,
                is_separator_regex=False
            )

            docs = st.session_state["store"]["docs"]
            contents = docs
            if docs and isinstance(docs[0], Document):
                contents = [doc.page_content for doc in docs]
            texts = text_splitter.create_documents(contents)
            n_chunks = len(texts)
            print(f"split into {n_chunks} chunks")

            # TODO 한국어 embedding 모델 잘하는 애를 쓸 수 있도록 변경
            if embedding_option == 'BM-K/KoSimCSE-roberta':
                embeddings = HuggingFaceEmbeddings(model_name="BM-K/KoSimCSE-roberta")
            elif embedding_option == 'text-embedding-3-small':
                embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"],
                                              model="text-embedding-3-small")

            proxy_embeddings = EmbeddingProxy(embeddings)
            # Create a vectorstore from documents
            # this will be a chroma collection with a default name.
            collection_name = st.session_state["store"]["collection_name"]
            persist_directory = os.path.join("./store", collection_name)
            db = Chroma(collection_name=collection_name,
                        embedding_function=proxy_embeddings,
                        persist_directory=persist_directory if os.path.exists(persist_directory) else None)
            db.add_documents(texts)

            st.session_state["store"]['retriever'] = db.as_retriever()

        retriever = st.session_state["store"].get('retriever')
        # 2. 프롬프트 생성
        if retriever:
            rag_prompt = hub.pull('rlm/rag-prompt')

            chain = (
                {
                    "context": RunnableLambda(get_question) | retriever | format_docs,
                    "question": RunnablePassthrough(),
                }
                | rag_prompt
                | llm
            )
        else:
            template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            prompt = PromptTemplate(
                input_variables=["question"],
                template=template,
            )

            # prompt = ChatPromptTemplate.from_messages(
            #     [
            #         # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
            #         MessagesPlaceholder(variable_name="history"),
            #         ("user", "{question}"),  # 사용자 질문을 입력
            #     ]
            # )

            chain = prompt | llm

        # chain_with_memory = (
        #     RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
        #         chain,  # 실행할 Runnable 객체
        #         get_session_history,  # 세션 기록을 가져오는 함수
        #         input_messages_key="question",  # 입력 메시지의 키
        #         history_messages_key="history",  # 기록 메시지의 키
        #     )
        # )

        response = chain.invoke(
            user_input,
            # 세션ID 설정
            config={"configurable": {"session_id": session_id}},
        )
        # st.session_state["messages"].append(ChatMessage(role="assistant", content=response))
