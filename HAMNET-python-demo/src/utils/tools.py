import tiktoken
import aiofiles
import hashlib
from typing import List
import re

ENCODER = tiktoken.get_encoding("cl100k_base")


def token_str(content: str) -> int:
    return len(ENCODER.encode(content))


async def get_pdf_md5(pdf_file_path: str) -> str:
    async with aiofiles.open(pdf_file_path, 'rb') as f:
        file_content = await f.read()
    pdf_md5 = hashlib.md5(file_content).hexdigest()
    return pdf_md5


def chunk_text_by_max_token(text: str, max_token=512) -> List[str]:
    if token_str(text) <= max_token:
        return [text]
    res = []
    num_split = token_str(text) // max_token
    # 切分的步长
    steps = [400, 200, 100]
    # 搜索标点符号正则
    punctuation = r'(?<!\d)([。？！…?!]|\.(?=\s))(?!\d)'
    while len(text) > 0:
        split_pos = 0
        while split_pos < len(text):
            for step in steps:
                next_step = min(step, len(text) - split_pos)
                temp_text = text[:split_pos + next_step]
                if token_str(temp_text) <= max_token:
                    split_pos += next_step
                    break
            else:
                break
        # 搜索最后一个标点符号
        search_res = re.search(punctuation, text[:split_pos][::-1])
        if search_res:
            split_pos = split_pos - int(search_res.end()) + 1
        res.append(text[:split_pos])
        text = text[split_pos:]
    return res

def chunk_list_by_max_token(paragraphs: List[str], max_token=512) -> List[str]:
    res = []
    temp_text = ''
    for paragraph in paragraphs:
        if token_str(paragraph) > max_token:
            res.extend(chunk_text_by_max_token(paragraph, max_token))
        elif token_str(temp_text + paragraph) > max_token:
            res.append(temp_text)
            temp_text = paragraph + '\n'
        else:
            temp_text += paragraph + '\n'
    return res

from langid.langid import LanguageIdentifier, model
from src.utils.tools import token_str
lang_identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
import uuid

def get_uuid()->str:
    return str(uuid.uuid4())

def detect_language(text: str) -> str:
    lang, _ = lang_identifier.classify(text)
    # print(lang)
    # chinese_characters = sum(1 for character in text if '\u4e00' <= character <= '\u9fff')
    # percentage = chinese_characters / len(text) * 100
    # if percentage > threshold:
    #     return 'zh'
    return lang


def is_english(text: str) -> bool:
    try:
        return detect_language(text) == 'en'
    except:
        return False
    
def check_language(language: str, text: str):
    if language != 'English' and is_english(text):
        raise Exception(f"The summary is not {language}")
    elif language == 'English' and not is_english(text):
        raise Exception(f"The summary is not {language}")