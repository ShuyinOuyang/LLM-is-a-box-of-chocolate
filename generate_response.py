import openai
import json
import asyncio
import os

import asyncgpt

async def get_response(name, prompt, bot, setting={}, previous_conversation=[], instruct_prompt=''):
    if len(previous_conversation) == 0:
        if instruct_prompt == '':
            instruct_prompt = 'Generate Python3 code (Markdown):\n'
        final_prompt = instruct_prompt + prompt
        message = [{"role": "user", "content": final_prompt}]
        if setting:
            temperature = setting['temperature']
            n = setting['topn']
        else:
            temperature = 0
            n = 1
        completion = await bot.chat_complete(message, temperature=temperature, n=n)
        res = {'name': name,
               'prompt': final_prompt,
               'completion': completion,
               'response': completion['choices'][0]['message']['content'],
               'message': message,
               'index': 0
               }
        return res
    elif len(previous_conversation) > 0:
        instruct_prompt = 'The last generated code is not good enough, please generate a better one.'
        message = []

        for conversation in previous_conversation:
            message.append({"role": "user", "content": conversation[name]['prompt']})
            message.append({"role": "assistant", "content": conversation[name]['response']})
        message.append({"role": "user", "content": instruct_prompt})
        completion = await bot.chat_complete(message, temperature=0)
        res = {'name': name,
               'prompt': instruct_prompt,
               'completion': completion,
               'response': completion['choices'][0]['message']['content'],
               'message': message,
               'index': 0
               }
        return res
    else:
        print('Reach the maximum loop threshold')
        return


async def openai_model_request(problem_dic, model, info_dic, setting={}, instruction_prompt=''):
    name_prompt_list = []
    for key in problem_dic:
        name_prompt_list.append([problem_dic[key]['name'], problem_dic[key]['description']])
    bot = asyncgpt.AsyncGPT(api_key=info_dic['api_key'], organization=info_dic['organization'], model=model)
    prompt_list = [get_response(name, prompt, bot, setting=setting, instruct_prompt=instruction_prompt) for name, prompt in name_prompt_list]
    res_list = await asyncio.gather(*prompt_list)
    return res_list

def get_problem_dic(dataset):
    problem_dic = {}
    if dataset == 'code_contest':
        with open('dataset/code_contests_test.json', 'r') as f:
            problem_list = json.load(f)
        for x in problem_list:
            problem_dic[x['name']] = x
    elif dataset == 'APPS':
        path = './APPS/test/'
        with open('log/major_revision_APPS_tag_list.json', 'r', encoding='utf-8') as f:
            tag_list = json.load(f)
        count = 0
        with open('log/dataset_APPS_model_gpt-3.5-turbo_topn_5_temperature_1.log_0', 'r') as f:
            for line in f.readlines():
                content = json.loads(line)
                if content['index'] == 0:
                    name = content['name']
                    with open(path + name + '/metadata.json', 'r', encoding='utf-8') as f:
                        json_dic = json.load(f)
                    with open(path + name + '/question.txt', 'r', encoding='utf-8') as f:
                        description = f.read()
                    problem_dic[name] = {
                        'name': name,
                        'difficulty': json_dic['difficulty'],
                        'tags': tag_list[count],
                        'description': description
                    }
                    count += 1
    elif dataset == 'HumanEval':
        problem_list = []
        with open('./HumanEval/HumanEval.jsonl', 'r') as f:
            for line in f.readlines():
                problem_list.append(json.loads(line))
                # break
        for x in problem_list:
            problem_dic[x['task_id']] = {
                'name': x['task_id'],
                'description': x['prompt'],
                'entry_point': x['entry_point']
            }
    return problem_dic

def generate_response_rq1_4(setting_dic):
    dataset = setting_dic['dataset']
    model = setting_dic['model']
    setting = setting_dic['setting']
    info_dic = setting_dic['info_dic']
    problem_dic = get_problem_dic(dataset)


    for sequence in range(1):
        log_file = './log/dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                   (dataset, model, setting['topn'], temperature, sequence)
        if os.path.exists(log_file):
            continue
        res_list = asyncio.run(openai_model_request(problem_dic, model, info_dic, setting=setting))
        for res in res_list:
            with open(log_file, 'a', encoding='utf-8') as f:
                json_str = json.dumps(res)
                f.write(json_str + '\n')

def generate_response_rq6(setting_dic):
    dataset = setting_dic['dataset']
    model = setting_dic['model']
    setting = setting_dic['setting']
    info_dic = setting_dic['info_dic']
    problem_dic = get_problem_dic(dataset)
    # complex prompt on randomness
    # 1. concise
    # instruction prompt = 'make the code as concise as possible'
    concise_request = True
    CoT_request = True
    # 1. concise
    if concise_request:
        instruct_prompt = 'Generate Python3 code (Markdown), make the code as concise as possible:\n'
        for sequence in range(5):
            log_file = './log/concise_dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                       (dataset, model, 1, temperature, sequence)
            if os.path.exists(log_file):
                continue
            res_list = asyncio.run(openai_model_request(problem_dic, model, info_dic, setting=setting))
            for res in res_list:
                with open(log_file, 'a', encoding='utf-8') as f:
                    json_str = json.dumps(res)
                    f.write(json_str + '\n')

    # 2. chain of thought
    # ask LLM to generate CoTs and then output the final code
    # instruction prompt = 'Generate Chain-of-Thought steps of how to solve the problem, and then generate Python3 code (Markdown):'
    if CoT_request:
        instruct_prompt = 'Generate Chain-of-Thought steps of how to solve the problem first, and then generate Python3 code (Markdown):\n'
        for sequence in range(5):
            log_file = './log/CoT_dataset_%s_model_%s_topn_%s_temperature_%s.log_%s' % \
                       (dataset, model, 1, temperature, sequence)
            if os.path.exists(log_file):
                continue
            res_list = asyncio.run(openai_model_request(problem_dic, model, info_dic, setting=setting))
            for res in res_list:
                with open(log_file, 'a', encoding='utf-8') as f:
                    json_str = json.dumps(res)
                    f.write(json_str + '\n')


if __name__ == "__main__":
    # load your openai account info
    with open('openai_info/info.json', 'r') as f:
        info_dic = json.load(f)
    # choose dataset
    dataset_ = ['code_contest', 'APPS', 'HumanEval']
    dataset = dataset_[1]
    # choose temperature
    temperature_ = [0, 1, 2]
    temperature = temperature_[0]
    # choose model
    model = 'gpt-3.5-turbo' # gpt-3-0125
    # model = 'gpt-4' # gpt-4-0613
    setting = {
        'temperature': temperature,
        'topn': 5 # the number of response API generated
    }

    setting_dic = {
        'dataset': dataset,
        'model': model,
        'setting': setting,
        'info_dic': info_dic
    }

    generate_response_rq1_4(setting_dic)
    generate_response_rq6(setting_dic)