import datetime, shutil
import logging
import os
from typing import Dict, Any, List
import openai
import jsonlines

TEST_RESULT = "./test_result/examples.txt"
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

def get_chat_response(title: str) -> str:
    """
    加入prompt话术范文写作，获取GPT-4模型的返回结果
    :param array:
    :param title: str
    :return:
    """
    global response
    openai.api_key = "your_key"

    # Make a request to the ChatGPT API
    messages = [{"role": "system", "content": "你是一个正在参加中国高考的考生，请基于用户输入的命题，用中文写出一篇800字左右的高考作文。 "
                                              "作文必须贴合主题，首尾呼应，结构匀称，立意明确，中心突出，感情真挚，语言流畅，意境深远， "
                                              "引经据典，善于运用修辞方法，构思精巧，见解新颖，具有积极作用。"},
                {"role": 'user', "content": title}]
    # print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Load {len(messages)} few-shot data.")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.75,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=0,
        )
    except Exception as e:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.75,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=1,
            presence_penalty=0,
        )
        logging.warning(e)
    finally:
        # Print the generated code
        print(response["choices"][0]["message"]['content'].strip())
    with jsonlines.open(TEST_RESULT + f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}" + ".jsonl",
                        mode='a') as writer:
        writer.write({"title": title, "essay": response["choices"][0]["message"]['content'].strip()})


def main():
    inputs = input("请输入高考作文题目：")
    get_chat_response(inputs)

if __name__ == "__main__":
    main()