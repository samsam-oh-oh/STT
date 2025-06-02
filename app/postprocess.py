from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def clean_transcription(raw_text: str) -> str:
    prompt = PromptTemplate.from_template(
        "너는 이제부터 문장 교정가이면서 IT 전문가야. 주어진 문장에서 **중복을 제거**하고, **문장을 교정**해줘.\n"
        "**단, 문장을 새로 생성하지 말고** 있는 문장 내에서 수정해줘.\n\n"
        "그리고 문장을 절대 추가하지마"
        "문장:\n{raw_text}"
    )

    chat = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY)
    chain = prompt | chat
    return chain.invoke({"raw_text": raw_text}).content