from typing import Dict, List, Tuple, Union
import json
import os
from src.api.openai_prompt import OpenAI
from src.utils.tools import get_pdf_md5
import asyncio
from dotenv import load_dotenv
from src.utils.tools import token_str
import re
# 读取 .env file.
load_dotenv()
api_keys = os.getenv('API_KEYS').split(',')
openai_prompt = OpenAI(api_keys=api_keys)


from src.utils.tools import chunk_text_by_max_token
async def novel_chapter_extract(
        semaphore,
        chapter: dict,
        attempts: int = 1,
        max_token: int = 12000
):
    if chapter.get('summary', '') != '':
        return chapter
    
    while attempts > 0:
        try:
            summary_before = ''
            if token_str(chapter['content']) > max_token:
                chunks = chunk_text_by_max_token(chapter['content'], max_token)
                for chunk in chunks:
                    res =  await openai_prompt.novel_chapter_extract_long(semaphore, chapter['title'], chunk, summary_before)
                    if res != '':
                        summary_before = res
            else:
                summary_before =  await openai_prompt.novel_chapter_extract(semaphore, chapter['title'], chapter['content'])
            chapter['summary'] = summary_before
            print(summary_before)
            return chapter
        except Exception as e:
            print(e)
            attempts -= 1
    chapter['summary'] = ''
    return chapter


async def novel_character_extract_sub(
        semaphore,
        title:str,
        text:str,
        attempts: int = 1,
):
    while attempts > 0:
        try:
            res =  await openai_prompt.novel_character_extract_thoughts(semaphore, title, text)
            print(res)
            if token_str(res) < 100:
                raise Exception('too short')
            return {
                'summary': res,
                'text': text
            }
        except Exception as e:
            print(e)
            attempts -= 1
    return {
        'summary': '',
        'text': text
    }

def value_process(value:str):
    if value.startswith("'"):
        value = value[1:]
    if value.endswith("'"):
        value = value[:-1]
    if value.startswith('[') and value.endswith(']'):
        temps = re.findall(r'\'(.*?)\'', value, re.DOTALL)
        value = temps
    return value

def json_to_embedding_chunk_cn(data):
    # 使用get方法提取主要元素，确保在缺少某些键的情况下不会出错
    date = data.get("Date", "")
    location = data.get("Location", "")
    event_summary = data.get("EventSummary", "")
    action = data.get("Details", {}).get("Action", "")
    dialogue = data.get("Details", {}).get("Dialogue", "")
    observations = data.get("Details", {}).get("Observations", "")
    emotional_response = data.get("EmotionalResponse", "")
    characters = "和".join(data.get("CharactersInvolved", []))
    impact = data.get("Impact", "")

    # 生成紧凑的chunk
    chunk = (f"{date} {location} {event_summary} "
             f"{action} {dialogue} {observations} "
             f"{emotional_response} {characters} {impact}").strip()

    return chunk

def parse_summary_to_json(summary: str) -> dict:
    summary = re.sub(r'\n\s*//.*\n', '', summary)
    pattern = r"'(.*?)': ('.*?'|\[.*?\]|\{.*?\}|\d+|true|false|null)"
    matches = re.findall(pattern, summary, re.DOTALL)
    new_res = []
    res = {}
    for match in matches:
        key, value = match
        value = value_process(value)
        if key=='Date' and res!= {}:
            new_res.append(res)
            res = {}
        elif key == 'Details':
            res_temp = {}
            temps = re.findall(pattern, value, re.DOTALL)
            for temp in temps:
                key_t, value_t = temp
                value_t = value_process(value_t)
                if type(value_t) == list:
                    value_t = '\n'.join(value_t)
                res_temp[key_t] = value_t
            value = res_temp
        res[key] = value
    new_res.append(res)
    return new_res

async def novel_character_extract(
        semaphore,
        chapter: dict,
        max_token: int = 8000,
):
    if len(chapter.get('plots', []))==0:
        chapter['plots']  = []
    flag = False
    if len(chapter['plots']):
        for plot in chapter['plots']:
            if type(plot['summary']) != str:
                plot['summary'] = ''
            parse_json = parse_summary_to_json(plot['summary'])
            plot['summary_json'] = parse_json
            embeddings = []
            for j in parse_json:
                embeddings.append(json_to_embedding_chunk_cn(j))
            plot['embeddings'] = embeddings
            if len(str(parse_json)) < 20:
                plot['summary'] = ''
            if plot.get('summary', '') == '':
                flag = True
                break
    if not flag and len(chapter['plots']) > 0:
        return chapter
    elif len(chapter['plots']) > 0:
        for plot in chapter['plots']:
            if plot.get('summary', '') == '':
                temp_res = await novel_character_extract_sub(semaphore,chapter['title'],plot['text'])
                plot['summary'] = temp_res['summary']
                print(plot['summary'])
        return chapter
    tasks = []
    chunks = chunk_text_by_max_token(chapter['content'], max_token)
    for chunk in chunks:
        tasks.append(novel_character_extract_sub(semaphore,chapter['title'],chunk))
    res = await asyncio.gather(*tasks)
    chapter['plots'] = res
    return chapter

async def novel_test(novel_dir:str='novels'):
    openai_prompt.openAIAPI.lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(10)
    novel_list = os.listdir(novel_dir)
    for novel in novel_list:
        if not novel.endswith('.json'):
            continue
        novel_path = os.path.join(novel_dir, novel)
        with open(novel_path, 'r') as f:
            data = json.load(f)
        tasks = []
        for chapter in data['chapter']:
            tasks.append(novel_character_extract(semaphore, chapter))
        results = await asyncio.gather(*tasks)
        data['chapter'] = results
        with open(novel_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def process_novel_txt(novel_dir = 'novels'):
    novel_list = os.listdir(novel_dir)
    for novel in novel_list:
        novel_path = os.path.join(novel_dir, novel)
        with open(novel_path, 'r') as f:
            content = f.read()
        # clean empty line
        content = re.sub(r'\n\s*\n', '\n', content)
        chapters = []
        for line in content.split('\n'):
            if line.startswith('　　'):
                if chapters[-1]['content']!= '':
                    chapters[-1]['content'] = chapters[-1]['content'] +'\n' + line.strip()
                else:
                    chapters[-1]['content'] = line.strip()
            else:
                chapters.append({'title': line.strip(), 'content': ''})
        final_res = []
        for chapter in chapters:
            if len(chapter['content'])>10:
                final_res.append(chapter)
        json_path = os.path.join(novel_dir, novel.replace('.txt', '.json'))
        res = {'title': novel.replace('.txt', '').strip(), 'chapter': final_res}
        with open(json_path, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        with open(novel_path, 'w') as f:
            f.write(content)

if __name__ == '__main__':
    asyncio.run(novel_test())