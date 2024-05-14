# LLM CHAT Demo #

## Requirements

To install requirements:

```
# Tuning 
pip install torch
pip install -r requirements.txt

# Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### .streamlit Folder 내부에 secrets.toml 파일 작성

프로젝트 루트에 secrets.toml 파일을 생성하고 아래와 같이 환경변수를 설정해주세요.

```
OPENAI_API_KEY={OpenAI API key}
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY={LangSmith 에서 API key 발급할 수 있음}
```

## Streamlit 실행

```
streamlit run main.py
```

## main.py 실행 모델 수정 방법

GGUF 모델은 ollama 로 서버 띄워서 실행

exl2 모델은 exllama-v2 로 모델 로드해서 실행

원본 모델 or AWQ or 16bit or fp8 로 실행하려면 vllm 서버 띄워서 실행

vllm 서버 띄우는 명령어 예시
```
python -m vllm.entrypoints.openai.api_server --model /home/mlteam/workspace/model_tuning/quantized_model/20240509/Llama-3-70B-Instruct-32k-cw_data_2040508-exl2/4.0bpw --dtype bfloat16 --api-key EMPTY
```

예시.
```python
import streamlit as st

llm_option = st.selectbox("Choose LLM",
                              (
                                  "Choose",
                                  "ChatGPT",
                                  "Llama-3-70B-Instruct-32k-cw_data_20240503",
                                  "Llama-3-70B-Instruct-32k-cw_data_20240503-Q4_K_M-GGUF",
                                  "Llama-3-70B-Instruct-32k-cw_data_20240508-Q3_K-GGUF",
                                  "Llama-3-70B-Instruct-32k-cw_data_2040508-4.0bpw-exl2",
                              ))

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
```
