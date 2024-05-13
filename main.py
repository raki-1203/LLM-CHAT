import os
import streamlit as st

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.llms.exllamav2 import ExLlamaV2
from langchain_community.llms.vllm import VLLMOpenAI, VLLM
from langchain_community.llms import OpenAI
from exllamav2.generator import ExLlamaV2Sampler

from utils import print_messages, StreamHandler

# API KEY 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

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
                                  "Choose",
                                  "ChatGPT",
                                  "Llama-3-70B-Instruct-32k-cw_data_20240503",
                                  "Llama-3-70B-Instruct-32k-cw_data_20240503-Q4_K_M-GGUF",
                                  "Llama-3-70B-Instruct-32k-cw_data_20240508-Q3_K-GGUF",
                                  "Llama-3-70B-Instruct-32k-cw_data_2040508-4.0bpw-exl2",
                              ))

    clear_btn = st.button("대화기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        del st.session_state["store"][session_id]
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
        elif llm_option == "Llama-3-70B-Instruct-32k-cw_data_2040508-4.0bpw-exl2":
            settings = ExLlamaV2Sampler.Settings(temperature=0.7,
                                                 top_k=-1,
                                                 top_p=0.8,
                                                 token_repetition_penalty=1.05)
            llm = ExLlamaV2(
                model_path="../model_tuning/quantized_model/20240509/Llama-3-70B-Instruct-32k-cw_data_2040508-exl2/4.0bpw",
                streaming=True,
                callbacks=[stream_handler],
                max_new_tokens=124,
                settings=settings,
                stop_sequences=["<|eot_id|>", "<|begin_of_text|>"]
            )
        elif llm_option == "Llama-3-70B-Instruct-32k-cw_data_20240503":
            # 따로 서버에서 vllm 모델을 실행해야 함
            llm = OpenAI(
                base_url="http://localhost:8000/v1",
                api_key="EMPTY",
                model_name="merged_output/20240509/Llama-3-70B-Instruct-32k-cw_data_20240508",
                callbacks=[stream_handler],
                streaming=True,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                temperature=0.8,
                top_p=0.8,
                max_tokens=2048,
            )

            # 이건 Streaming 기능이 안되는 듯
            # llm = VLLM(
            #     model="../model_tuning/merged_output/20240509/Llama-3-70B-Instruct-32k-cw_data_20240508",
            #     tensor_parallel_size=4,
            #     max_new_tokens=2048,
            #     callbacks=[stream_handler],
            #     streaming=True,
            # )

        # 2. 프롬프트 생성
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
        chain_with_memory = (
            RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
                chain,  # 실행할 Runnable 객체
                get_session_history,  # 세션 기록을 가져오는 함수
                input_messages_key="question",  # 입력 메시지의 키
                history_messages_key="history",  # 기록 메시지의 키
            )
        )

        response = chain_with_memory.invoke(
            {"question": user_input},
            # 세션ID 설정
            config={"configurable": {"session_id": session_id}},
        )
        # st.session_state["messages"].append(ChatMessage(role="assistant", content=response))
