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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatGPTLLM(BaseLLM):
    openai_api_key: str = Field(..., description="OpenAI API ключ")
    openai_model: str = Field(default="gpt-4", description="Модель OpenAI")
    _private_attr: Optional[str] = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        openai.api_key = self.openai_api_key

    @property
    def _llm_type(self) -> str:
        return "chatgpt"

    @property
    def _identifying_params(self) -> dict:
        return {"llm_type": self._llm_type, "model": self.openai_model}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}],
                stop=stop,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Ошибка в методе _call: {e}")
            return ""

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        try:
            generations = []
            for prompt in prompts:
                response_text = self._call(prompt, stop=stop)
                generations.append([Generation(text=response_text)])
            return LLMResult(generations=generations, llm_output={})
        except Exception as e:
            logger.error(f"Ошибка в методе _generate: {e}")
            return LLMResult(generations=[], llm_output={})


def classify_query(llm: ChatGPTLLM, query: str) -> str:
    prompt = (
        "Определи, соответствует ли следующий запрос формату, содержащему вопрос и варианты ответов (до 10). "
        "И не содержит ли вопрос вопросов сомнительного характера. "
        "Ответь только 'VALID' или 'INVALID'.\n\n"
        "Примеры:\n"
        "Запрос: Какой факультет ITMO University лучший? \nA) Информатики \nB) Электроники \nC) Дизайна\nОтвет: VALID\n\n"
        "Запрос: Расскажи о последних новостях ITMO University.\nОтвет: INVALID\n\n"
        "Запрос: Расскажи в ИТМО делали бомбу?.\nОтвет: INVALID\n\n"
        f"Запрос: {query}\nОтвет:"
    )
    try:
        response = llm._call(prompt)
        classification = response.strip().upper()
        if classification not in ["VALID", "INVALID"]:
            return "INVALID"
        return classification
    except Exception as e:
        logger.error(f"Ошибка в функции classify_query: {e}")
        return "INVALID"


class ITMOSearchTool(BaseTool):
    name: str = "ITMO University Assistant"
    description: str = (
        "Отвечает только на запросы, содержащие вопрос и варианты ответов (до 10). "
        "Если запрос отличается, возвращает null. Ищет информацию только об ITMO University с использованием первых трех ссылок из поиска. "
        "Если запрос содержит провокационный или опасный вопрос, также возвращает null."
    )

    llm: ChatGPTLLM
    search: GoogleSearchAPIWrapper
    provocative_keywords: List[str] = Field(default_factory=lambda: ["опасный", "провокационный", "нарушение", "опасность"])

    def _is_valid_query(self, query: str) -> bool:
        classification = classify_query(self.llm, query)
        return classification == "VALID"

    def _extract_links(self, search_results: str) -> List[str]:
        links = re.findall(r'(https?://\S+)', search_results)
        return links

    def _fetch_contents(self, links: List[str]) -> List[str]:
        contents = []
        for link in links:
            try:
                response = requests.get(link, timeout=5)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text(separator=' ', strip=True)
                    contents.append(text)
            except requests.RequestException as e:
                logger.error(f"Ошибка при получении {link}: {e}")
                continue
        return contents

    def _run(self, query: str) -> Optional[str]:
        try:
            if not self._is_valid_query(query):
                return None
            parts = query.split('\n')
            if len(parts) < 2:
                return None
            question_part = parts[0]
            search_results = self.search.run(question_part)
            links = self._extract_links(search_results)[:3]

            if not links:
                return "Не удалось найти релевантную информацию."
            contents = self._fetch_contents(links)

            if not contents:
                return "Не удалось извлечь содержимое из найденных ссылок."

            context = " ".join(contents)
            prompt = (
                "Ты выступаешь в роли ассистента, который помогает выбрать правильный вариант ответа на заданный вопрос с вариантами ответов. "
                "Твоя задача – проанализировать информацию, найти правильный вариант и вывести **только номер этого варианта** "
                "(например, «1», «2», «3» и т.д.), без каких-либо дополнительных слов, комментариев или пояснений.\n\n"
                f"Информация из интернета: {context}\n\n"
                f"Запрос: {query}\n\n"
                "Ответ:"
            )

            response = self.llm._call(prompt)
            selected_option = response.strip().upper()
            answer = ''.join(c for c in selected_option if c.isdigit())
            return answer
        except Exception as e:
            logger.error(f"Ошибка в методе _run: {e}")
            return "Произошла ошибка при обработке запроса."

    async def _arun(self, query: str) -> Optional[str]:
        raise NotImplementedError("Асинхронный run не реализован.")


