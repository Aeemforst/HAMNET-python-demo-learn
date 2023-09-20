# -*- coding: utf-8 -*-
import re
from src.api.openai_api import openAI
import re
import asyncio
from src.utils.tools import token_str, is_english, get_uuid, check_language


def clean_section_text(text: str) -> str:
    final_res = ''
    res_lines = text.split('\n')
    if len(res_lines) > 1:
        for line in res_lines:
            if re.match(r"^\((.*)\)$", line, re.DOTALL):  # 清洗像(note: xxx)等无关内容
                continue
            if token_str(line) > 50:  # 清洗太短的行,提取出主要部分
                final_res += line + '\n'
    else:
        final_res = text
    return str(final_res)


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


class OpenAI:

    def __init__(
        self,
        api_keys: list = [],
        model_name="gpt-3.5-turbo-0613",
        p=1.0,
        temperature=1.0,
    ):
        self.openAIAPI = openAI(api_keys=api_keys,
                                model_name=model_name,
                                top_p=p,
                                temperature=temperature,
                                apiTimeInterval=0.1)

    def print_token(self, result):
        print("Prompt used: ", str(result[1]) + " tokens.")
        print("Completion used: ", str(result[2]) + " tokens.")
        print("Totally used: ", str(result[3]) + " tokens.")


    async def translate_output(self, semaphore, text: str, language: str):
        async with semaphore:
            print(f"translate {language}")
            convo_id = "translate" + str(get_uuid())
            self.openAIAPI.reset(convo_id=convo_id,
                                 system_prompt=f"你是一个学术翻译专家")
            content = f"""
            {text}
            将上面文本翻译为{language}。请注意，原文中的所有学术词汇和缩写应保留为英文，其余部分请使用{language}进行翻译。除翻译内容外，不需要输出任何其他信息, 保留原来的文本格式。
            """
            result = await self.openAIAPI.ask(prompt=content,
                                              role="user",
                                              convo_id=convo_id)
            self.openAIAPI.conversation[convo_id] = None
            res = str(result[0])
            print('translate done')
            # 断言输出语言是否符合预期
            if language != 'English' and is_english(res):
                raise Exception(f"The summary is not {language}")
            return res

    async def optimize_query(
        self,
        semaphore,
        query: str,
        language: str = 'English',
    ):
        async with semaphore:
            convo_id = "summary" + str(get_uuid())
            self.openAIAPI.reset(convo_id=convo_id, system_prompt=f"")
            content = f"Translate the user's question into {language}, and then rephrase it into an exact, standardized query specifically designed for vector database search. Focus on retaining the essential elements and core concepts of the original question. The restructured query must be formulated to maximize the probability of a precise match in a vectorized text database. Output only the final and most accurately optimized query:{query}"
            result = await self.openAIAPI.ask(prompt=content,
                                              role="user",
                                              convo_id=convo_id)
            self.openAIAPI.conversation[convo_id] = None

            return str(result[0])

    async def novel_chapter_extract(
        self,
        semaphore,
        title: str,
        text: str,
    ):
        async with semaphore:
            convo_id = "summary" + str(get_uuid())
            self.openAIAPI.reset(convo_id=convo_id, system_prompt=f"")
            content = f"""
请根据以下指导，对提供的小说章节内容进行深入分析，并按照给定的格式为每个要素提供详细的信息:
- # 人物:列出小说中的主要角色和重要的配角。 
- # 情节:描述章节中的核心事件或活动，涉及的主要冲突、转折或决定性时刻。 
- # 场景:识别和描述故事主要发生的地点或背景。
- # 视角:确定章节或故事的叙述视角，如第一人称、第三人称等。
- # 主题:总结章节或故事的核心思想、信息或主题。
- # 风格:描述文本的写作风格，是否现实、抽象、诗意、寓言性等。

例子:
小说章节内容:

在一个寂静的古老村庄，李白醉酒地走在石子路上，忽然停下，抬头仰望星空，并开始吟咏。杜甫和王之涣站在一旁，低声讨论着李白的诗才。夜晚的寂静只被他们的声音所打破，月光洒在古老的石屋和路上。

---

# 人物:
- 主要角色:李白
- 配角:杜甫、王之涣

# 情节:
- 李白醉酒地走在石子路上，停下来仰望星空并吟咏。
- 杜甫和王之涣讨论李白的诗才。

# 场景:
- 场景1:寂静的古老村庄的石子路上
- 场景2:夜晚，月光下的古老石屋

# 视角:
- 第三人称

# 主题:
- 对自然与诗歌的赞美、友人间的尊重和欣赏

# 风格:
- 抒情与描写，强调情感和环境

---

下面是你需要提炼的小说章节的部分内容:
{title}
{text}
开始提炼:
            """
            result = await self.openAIAPI.ask(prompt=content,
                                              role="user",
                                              convo_id=convo_id)
            self.openAIAPI.conversation[convo_id] = None
            return str(result[0])

    async def novel_chapter_extract_long(
        self,
        semaphore,
        title: str,
        text: str,
        summary_before: str,
    ):
        async with semaphore:
            convo_id = "summary" + str(get_uuid())
            self.openAIAPI.reset(convo_id=convo_id, system_prompt=f"")
            content = f"""
请根据以下指导，对提供的小说章节内容进行深入分析，并按照给定的格式为每个要素提供详细的信息:
- # 人物:列出小说中的主要角色和重要的配角。 
- # 情节:描述章节中的核心事件或活动，涉及的主要冲突、转折或决定性时刻。 
- # 场景:识别和描述故事主要发生的地点或背景。
- # 视角:确定章节或故事的叙述视角，如第一人称、第三人称等。
- # 主题:总结章节或故事的核心思想、信息或主题。
- # 风格:描述文本的写作风格，是否现实、抽象、诗意、寓言性等。

例子:
小说章节内容:

在一个寂静的古老村庄，李白醉酒地走在石子路上，忽然停下，抬头仰望星空，并开始吟咏。杜甫和王之涣站在一旁，低声讨论着李白的诗才。夜晚的寂静只被他们的声音所打破，月光洒在古老的石屋和路上。

---

# 人物:
- 主要角色:李白
- 配角:杜甫、王之涣

# 情节:
- 李白醉酒地走在石子路上，停下来仰望星空并吟咏。
- 杜甫和王之涣讨论李白的诗才。

# 场景:
- 场景1:寂静的古老村庄的石子路上
- 场景2:夜晚，月光下的古老石屋

# 视角:
- 第三人称

# 主题:
- 对自然与诗歌的赞美、友人间的尊重和欣赏

# 风格:
- 抒情与描写，强调情感和环境

---

下面是你需要提炼的小说章节的部分内容:
{title}
{text}

---

这是之前的要素提炼, 请在此基础上继续提炼:
{summary_before}

开始提炼:
            """
            result = await self.openAIAPI.ask(prompt=content,
                                              role="user",
                                              convo_id=convo_id)
            self.openAIAPI.conversation[convo_id] = None
            return str(result[0])

    async def novel_character_extract_dialogue(
        self,
        semaphore,
        title: str,
        text: str,
        character: str,
        summary_before: str = '',
    ):
        async with semaphore:
            convo_id = "summary" + str(get_uuid())
            self.openAIAPI.reset(convo_id=convo_id, system_prompt=f"")
            content = f"""
**请根据以下精细化指南，分析给定的小说章节内容，并精确地提炼出特定角色的所有对话**：

**详细指南**：

- **目标角色**：{character}

- **提取步骤**：
  1. **初始化分析**：首先全文浏览一遍，确定{character}是否在文本中出现。
  2. **角色对话定位**：找到{character}首次出现的地方，从该点开始逐句检查直到文本结束。
  3. **对话提取**：每当遇到{character}的名字和一个引号，标记起点。从这个起点开始，提取对话内容直到引号闭合。
  4. **其他角色的对话**：当提取{character}的对话时，也应查找其前后与其他角色的对话，以确保对话上下文完整。
  5. **确认角色身份**：确保提取的对话前的角色名称与{character}完全匹配，并确认该对话内容确实是{character}说的。
  6. **异常处理**：如果文本中未提及{character}或其对话，直接返回“无”。

- **格式规范**：
  1. **对话标记**：使用双引号“ ”标记对话内容。
  2. **角色名格式**：角色名称与文本中的一致，并紧跟冒号和对话内容。
  3. **对话顺序**：根据在文本中的出现顺序，逐一列出对话。

**输出格式**：
- [其他角色名]：“[对话内容]”
- {character}：“[对话内容]”
...

**给定文本**：
{title}
{text}

**之前的提炼内容**：
{summary_before}

**现在，请开始提炼**：
            """
            result = await self.openAIAPI.ask(prompt=content,
                                              role="user",
                                              convo_id=convo_id)
            self.openAIAPI.conversation[convo_id] = None
            return str(result[0])

#     async def novel_character_extract_thoughts(
#             self,
#             semaphore,
#             title:str,
#             text:str,
#             character:str,
#             summary_before:str = '',
#     ):
#         async with semaphore:
#             character = '叶文洁'
#             convo_id = "summary" + str(get_uuid())
#             self.openAIAPI.reset(convo_id=convo_id, system_prompt=f"""你是一个演员, 你需要扮演角色叶文洁, 她的基本信息如下: 叶文洁，女，出生于1947年6月，清华大学天体物理学硕士学位。""")
#             content = f"""
# 开始角色扮演任务:
# 任务步骤：
# 1. 文本扫描：快速浏览文本，判断其中是否存在关于“叶文洁”的描述或提及。
# 2. 判断情境：如果文中有关于叶文洁的描述，判定这些描述是否充分明确以供扮演。如果是，则继续步骤3；如果不是，直接输出无法扮演的原因。
# 3. 转化：
#    - 将与叶文洁相关的第三人称描述或对话转化为第一人称。
#    - 描述行为时，如“叶文洁看向窗外”，转为“我看向窗外”。
#    - 描述情感或思考时，如“叶文洁感到迷茫”，转为“我感到迷茫”。
#    - 直接提取叶文洁的对话，如“‘为什么?’ 叶文洁问。”，转为“我问：‘为什么?’”。
# 4. 组织输出：根据时间和情境顺序组织扮演的描述。

# 例子1:
# Input: `叶文洁看到的砍伐只能用疯狂来形容，高大挺拔的兴安岭落叶松、四季长青的樟子松、亭亭玉立的白桦、耸入云天的山杨、西伯利亚冷杉，以及黑桦、柞树、山榆、水曲柳、钻天柳、蒙古栎，见什么伐什么，几百把油锯如同一群钢铁蝗虫，她的连队所过之处，只剩下一片树桩。`
# Thought: 提供的内容大致讲了叶文洁看到的砍伐场景, 通过Input的信息我能够在这些场景中扮演叶文洁. 所以我现在开始扮演叶文洁, 我就是叶文洁。
# Output: 在砍伐场景中，我看到了很多树木都被砍伐, 我的连队所过之处，只剩下一片树桩

# 例子2:
# Input: `汪淼抬头看看反应黑箱，觉得它像一个子宫，工程师们正围着它忙碌，艰难地维持着正常的运行。在这场景前面，叠现着幽灵倒计时。`
# Thought: 提供的内容大致讲汪淼看反应黑箱, 基于提供的的Input, 我找不到足够的信息来扮演叶文洁。原因是没有叶文洁的信息。
# output: 无法扮演，因为没有叶文洁的信息。
# ---
# Begin!
# Input: `{title} {text}`
#             """
#             result = await self.openAIAPI.ask(prompt=content,
#                                               role="user",
#                                               convo_id=convo_id)
#             self.openAIAPI.conversation[convo_id] = None
#             return str(result[0])

#     async def novel_character_extract_thoughts(
#             self,
#             semaphore,
#             title:str,
#             text:str,
#             character:str,
#             summary_before:str = '',
#     ):
#         async with semaphore:
#             character = '叶文洁'
#             convo_id = "summary" + str(get_uuid())
#             self.openAIAPI.reset(convo_id=convo_id, system_prompt=f"""从以下小说段落中，请根据以下指导，只提取并拓展关于角色"叶文洁"的记忆。如果小说段落中没有提及此角色，请回复"该段落中没有提及指定角色的信息。""")
#             content = f"""
# 任务步骤：
# 1. 文本扫描：快速浏览文本，判断其中是否存在关于“叶文洁”的描述或提及。
# 2. 判断情境：如果文中有关于叶文洁的描述，判定这些描述是否充分明确以供扮演。如果是，则继续步骤3；如果不是，直接输出无法扮演的原因。
# 3. 转化：
#    - 将与叶文洁相关的第三人称描述或对话转化为第一人称。
#    - 描述行为时，如“叶文洁看向窗外”，转为“我看向窗外”。
#    - 描述情感或思考时，如“叶文洁感到迷茫”，转为“我感到迷茫”。
#    - 直接提取叶文洁的对话，如“‘为什么?’ 叶文洁问。”，转为“我问：‘为什么?’”。
# 4. 组织输出：根据时间和情境顺序组织扮演的描述。
# 例子1:
# Input: `叶文洁看到的砍伐只能用疯狂来形容，高大挺拔的兴安岭落叶松、四季长青的樟子松、亭亭玉立的白桦、耸入云天的山杨、西伯利亚冷杉，以及黑桦、柞树、山榆、水曲柳、钻天柳、蒙古栎，见什么伐什么，几百把油锯如同一群钢铁蝗虫，她的连队所过之处，只剩下一片树桩。`
# Thought: 提供的内容大致讲了叶文洁看到的砍伐场景, 通过Input的信息我能够在这些场景中扮演叶文洁. 所以我现在开始扮演叶文洁, 我就是叶文洁。
# Output: 在砍伐场景中，我看到了很多树木都被砍伐, 我的连队所过之处，只剩下一片树桩
# 例子2:
# Input: `汪淼抬头看看反应黑箱，觉得它像一个子宫，工程师们正围着它忙碌，艰难地维持着正常的运行。在这场景前面，叠现着幽灵倒计时。`
# Thought: 提供的内容大致讲汪淼看反应黑箱, 基于提供的的Input, 我找不到足够的信息来扮演叶文洁。原因是没有叶文洁的信息。
# output: 该段落中没有提及指定角色的信息。
# ---
# Begin!
# Input: {title} {text}
#             """
#             result = await self.openAIAPI.ask(prompt=content,
#                                               role="user",
#                                               convo_id=convo_id)
#             self.openAIAPI.conversation[convo_id] = None
#             return str(result[0])

    async def novel_character_extract_thoughts(
        self,
        semaphore,
        title: str,
        text: str,
    ):
        async with semaphore:
            character = '叶文洁'
            convo_id = "summary" + str(get_uuid())
            self.openAIAPI.reset(
                convo_id=convo_id,
                system_prompt=f"""从以下小说段落中，请根据以下指导，只提取并拓展关于角色的描述。""")
            content = f"""
根据以下提供的小说文本，请为主要角色提取并按照指定的格式归纳其记忆库：
{title} {text}
"""
            content += """
格式如下：
[
    {
        'Date': '例如：2023年春',
        'Location': '详细地点或场景',
        'EventSummary': '简要概述',
        'Details': {
            'Action': '主要行为或动作',
            'Dialogue': '关键对话或交流',
            'Observations': '其他值得注意的细节或背景信息'
        },
        'EmotionalResponse': '角色的主要情感或反应',
        'CharactersInvolved': ['人物A', '人物B'],
        'Impact': '事件对情节进展或角色发展的直接或潜在影响'
    },
    // ... 其他记忆条目
]

"""

            result = await self.openAIAPI.ask(prompt=content,
                                              role="user",
                                              convo_id=convo_id)
            self.openAIAPI.conversation[convo_id] = None
            return str(result[0])

