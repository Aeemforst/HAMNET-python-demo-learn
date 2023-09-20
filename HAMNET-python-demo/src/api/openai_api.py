import time
import requests
from queue import PriorityQueue as PQ
import time
import httpx
import asyncio
import re
import json
from src.utils.tools import token_str


def get_content(data_str):
    try:
        encoded_data = data_str.encode('utf-8')
        encoded_data = encoded_data[6:]
        data = json.loads(encoded_data)
    except Exception as e:
        return None
    choices = data.get('choices')  # 获取"choices"的值
    # 检查 "choices" 是否是一个列表，且列表是否非空
    if isinstance(choices, list) and len(choices) > 0:
        # 获取第一个 "choice" 的 "content" 值
        content = choices[0].get('delta').get('content')
        return content
    else:
        return None


def truncate_text(text: str, max_token: int = 15000) -> str:
    token_cost = token_str(text)
    if token_cost > max_token:
        while token_cost > max_token:
            char_to_truncate = int((token_cost - max_token))
            text = text[:-char_to_truncate]
            token_cost = token_str(text)
        text = text + "..."
    else:
        return text


content_pattern = re.compile(r'"content":"([^"]*)"')


class openAI:
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_keys: list,
        proxy=None,
        max_tokens: int = 16000,
        temperature: float = 0,
        top_p: float = 1.0,
        model_name: str = "gpt-3.5-turbo-0613",
        reply_count: int = 1,
        system_prompt="You are a smart AI assistant.",
        lastAPICallTime=time.time() - 100,
        apiTimeInterval: float = 20.0,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.apiTimeInterval = apiTimeInterval
        self.session = requests.Session()
        self.api_keys = PQ()
        self.trash_api_keys = PQ()
        for key in api_keys:
            self.api_keys.put((lastAPICallTime, key))
        self.proxy = proxy
        if self.proxy:
            proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            self.session.proxies = proxies
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reply_count = reply_count
        self.conversation = {}
        if token_str(self.system_prompt) > 1000:
            raise Exception("System prompt is too long")

    # async def init_lock(self):
    #     self.lock = asyncio.Lock()

    async def get_api_key(self):
        async with self.lock:
            while self.trash_api_keys.qsize() and self.api_keys.qsize():
                trash_key = self.trash_api_keys.get_nowait()
                api_key = self.api_keys.get_nowait()

                if trash_key[1] == api_key[1]:
                    self.api_keys.put_nowait(
                        (time.time() + 24 * 3600, api_key[1]))
                else:
                    self.trash_api_keys.put_nowait(trash_key)
                    self.api_keys.put_nowait(api_key)
                    break

            api_key = self.api_keys.get_nowait()
            if api_key[0] > time.time():
                print('API key exhausted')
                raise Exception('API key Exhausted')

            delay = await self._calculate_delay(api_key)
            await asyncio.sleep(delay=delay)
            self.api_keys.put_nowait((time.time(), api_key[1]))

        return api_key[1]

    async def _calculate_delay(self, apiKey):
        elapsed_time = time.time() - apiKey[0]
        if elapsed_time < self.apiTimeInterval:
            return self.apiTimeInterval - elapsed_time
        else:
            return 0

    def add_to_conversation(self,
                            message: str,
                            role: str,
                            convo_id: str = "default"):
        if (convo_id not in self.conversation):
            self.reset(convo_id)
        self.conversation[convo_id].append({"role": role, "content": message})

    def __truncate_conversation(self, convo_id: str = "default"):
        """
        Truncate the conversation
        """
        last_dialog = self.conversation[convo_id][-1]
        query = str(last_dialog['content'])
        query = truncate_text(query, self.max_tokens)
        self.conversation[convo_id] = self.conversation[convo_id][:-1]
        conversation_token_costs = sum(
            [token_str(x["content"]) for x in self.conversation[convo_id]])
        total_token_cost = conversation_token_costs + token_str(query)

        while total_token_cost > self.max_tokens:
            farthest_dialogue = self.conversation[convo_id].pop(1)
            total_token_cost -= token_str(farthest_dialogue['content'])

        last_dialog['content'] = query
        self.conversation[convo_id].append(last_dialog)

    async def ask_stream(self,
                         prompt: str,
                         role: str = "user",
                         convo_id: str = "default",
                         **kwargs) -> str:
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id)
        self.add_to_conversation(prompt, "user", convo_id=convo_id)
        self.__truncate_conversation(convo_id=convo_id)
        if self.token_cost(convo_id=convo_id) > 3000:
            model_name = 'gpt-3.5-turbo-16k-0613'
        else:
            model_name = 'gpt-3.5-turbo-0613'
        apiKey = await self.get_api_key()
        async with httpx.AsyncClient() as client:
            async with client.stream(
                    method="POST",
                    url="https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization":
                        f"Bearer {kwargs.get('api_key', apiKey)}"
                    },
                    json={
                        "model": model_name,
                        "messages": self.conversation[convo_id],
                        # kwargs
                        "temperature": kwargs.get('temperature',
                                                  self.temperature),
                        "top_p": kwargs.get('top_p', self.top_p),
                        "n": self.reply_count,
                        "user": role,
                        "stream": True,
                    },
            ) as response:
                if response.status_code != 200:
                    if response.status_code == 403:  # API key error
                        self.trash_api_keys.put((time.time(), apiKey))
                    raise Exception(f"Error: {response.status_code}")
                async for chunk in response.aiter_lines():
                    if chunk:
                        res = get_content(chunk)
                        if res:
                            if kwargs.get('test', False):
                                print(str(res), end='')
                            yield str(res)

    from typing import Tuple

    async def ask(self,
                  prompt: str,
                  role: str = "user",
                  convo_id: str = "default",
                  **kwargs) -> Tuple[str, int, int, int]:
        """
        Non-streaming ask
        """
        full_response = ''
        async for response in self.ask_stream(prompt, role, convo_id,
                                              **kwargs):
            full_response += response
        # full_response = full_response.replace(r"\\\"", r"\\\\")
        prompt_token = self.token_cost(convo_id=convo_id)
        completion_token = token_str(full_response)
        self.add_to_conversation(full_response, role, convo_id=convo_id)
        total_token = prompt_token + completion_token
        return full_response, prompt_token, completion_token, total_token

    def reset(self, convo_id: str = "default", system_prompt=None):
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {
                "role": "system",
                "content": str(system_prompt or self.system_prompt)
            },
        ]
        
    def token_cost(self, convo_id: str = "default"):
        return token_str("\n".join(
            [x["content"] for x in self.conversation[convo_id]]))

def main():
    return
