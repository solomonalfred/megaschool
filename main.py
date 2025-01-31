import time
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from request import PredictionRequest, PredictionResponse
from logger import setup_logger
from agent import *
import os
import re
import json
import requests
import logging
from typing import Optional, List

import openai
from pydantic import BaseModel, Field, PrivateAttr
from langchain.llms.base import BaseLLM
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.schema import Generation, LLMResult

from bs4 import BeautifulSoup



app = FastAPI()
logger = None

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
  raise ValueError("GOOGLE_API_KEY и GOOGLE_CSE_ID должны быть установлены в переменных окружения.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
  raise ValueError("OPENAI_API_KEY не установлен в переменных окружения.")

llm = ChatGPTLLM(openai_api_key=OPENAI_API_KEY)
search = GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id=GOOGLE_CSE_ID
)
itmo_tool = ITMOSearchTool(llm=llm, search=search)
tools = [itmo_tool]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def process_request(query: str, req_id: str) -> dict:
    if classify_query(llm, query) != "VALID":
        return {
            "id": req_id,
            "answer": None,
            "reasoning": "Запрос не соответствует необходимому формату или содержит недопустимые элементы.",
            "sites": []
        }
    question_part = query.split('\n')[0]
    search_results = search.run(question_part)
    used_sites = itmo_tool._extract_links(search_results)
    used_sites = used_sites[:3]
    contents = itmo_tool._fetch_contents(used_sites)
    context = " ".join(contents)
    prompt = (
        "Ты выступаешь в роли ассистента, который помогает выбрать правильный вариант ответа на заданный вопрос с вариантами ответов. \n"
        "Твоя задача – проанализировать предоставленную информацию, выбрать правильный вариант ответа и объяснить, почему именно он верный.\n"
        "Выведи результат в формате JSON с двумя полями:\n"
        "  - 'answer': номер правильного варианта (например, 1, 2, 3 и т.д.)\n"
        "  - 'reasoning': краткое объяснение выбора\n\n"
        f"Информация из интернета: {context}\n\n"
        f"Запрос: {query}\n\n"
        "Ответ в формате JSON:"
    )

    response = llm._call(prompt)
    try:
        result = json.loads(response)
        answer = result.get("answer", "").strip()
        reasoning = result.get("reasoning", "").strip()
    except Exception as e:
        logger.error(f"Ошибка при разборе ответа LLM: {e}")
        answer = response.strip()
        reasoning = "Не удалось получить объяснение из ответа."

    return {
        "id": req_id,
        "answer": answer,
        "reasoning": reasoning,
        "sites": used_sites
    }



@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")

        res = process_request(body.query, str(body.id))

        response = PredictionResponse(
            id=res['id'],
            answer=res['answer'],
            reasoning=res['reasoning'],
            sources=res['sources'],
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
