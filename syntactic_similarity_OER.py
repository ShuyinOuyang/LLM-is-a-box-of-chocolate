import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import nltk
import os
import tokenize
import scipy.stats as stats
from nltk.translate.bleu_score import sentence_bleu
from difflib import SequenceMatcher

# def analyze_among_top0_5(dataset, model, temperature):
def analyze_among_top0_5(file):
    dataset, model, temperature = '', '', ''
    file_str = file.split('/')[-1]
    file_suffix = file_str.split('.')[-1]
    file_prefix = file_str.replace('.'+file_suffix, '')
    file_prefix_str_list = file_prefix.split('_')
    type = file_prefix_str_list[0]
    for i, piece in enumerate(file_prefix_str_list):
        if piece == 'dataset':
            dataset = file_prefix_str_list[i+1]
        elif piece == 'model':
            model = file_prefix_str_list[i+1]
        elif piece == 'temperature':
            temperature = file_prefix_str_list[i+1]
    if 'topn_5' in file:
        type = 'old'
    if dataset == 'code':
        dataset = 'code_contest'
    if type != 'dataset':
        save_dir = './syntactic_similarity/%s_%s_%s_%s/' % (type, dataset, model, temperature)
    else:
        save_dir = './syntactic_similarity/%s_%s_%s/' % (dataset, model, temperature)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def syntatic_similarity(problem_dic, name, code_candidates, case_status_list):
        same_output_between_5 = []
        same_output_between_5_correct = []
        same_output_between_5_timeout = []
        same_output_between_5_exception = []
        same_output_between_5_execution_error = []
        Levenshtein_edit_distance = []
        LED_all = []
        for i in range(len(case_status_list[0])):
            output_set = set()
            for j in range(len(problem_dic[name]['code_candidates'])):
                output_set.add(case_status_list[j][i])
            # print(output_set)
            if len(output_set) == 1:
                same_output_between_5.append(i)
                if list(output_set)[0] == 'timeout':
                    same_output_between_5_timeout.append(i)
                elif 'execution error' in list(output_set)[0]:
                    same_output_between_5_execution_error.append(i)
                elif list(output_set)[0] == 'exception' or sum([len(case) for case in code_candidates]) == 0:
                    same_output_between_5_exception.append(i)
                else:
                    same_output_between_5_correct.append(i)

        for i in range(len(problem_dic[name]['code_candidates'])):
            if i == 0:
                continue
            Levenshtein_edit_distance.append(nltk.edit_distance(code_candidates[0], code_candidates[i]))
        for i in range(len(problem_dic[name]['code_candidates'])):
            for j in range(i+1, len(problem_dic[name]['code_candidates'])):
                LED_all.append(nltk.edit_distance(code_candidates[i], code_candidates[j]))

        problem_dic[name]['syntatic_similarity'] = {
            'same_output_between_5': same_output_between_5,
            'same_output_between_5_correct': same_output_between_5_correct,
            'same_output_between_5_timeout': same_output_between_5_timeout,
            'same_output_between_5_exception': same_output_between_5_exception,
            'same_output_between_5_execution_error': same_output_between_5_execution_error,
            'Levenshtein_edit_distance': Levenshtein_edit_distance,
            'LED_all': LED_all
        }
        total_test_case_num = len(problem_dic[name]['code_candidates'][0]['case_status'])
        if total_test_case_num == 0:
            syntatic_similarity_res = {
                'same_output_between_5': 0,
                'same_output_between_5_correct': 0,
                'same_output_between_5_timeout': 0,
                'same_output_between_5_exception': 0,
                'same_output_between_5_execution_error': 0,
                'Levenshtein_edit_distance': Levenshtein_edit_distance,
                'LED_all': LED_all
            }
        else:
            syntatic_similarity_res = {
                'same_output_between_5': len(problem_dic[name]['syntatic_similarity']['same_output_between_5']) / total_test_case_num,
                'same_output_between_5_correct': len(problem_dic[name]['syntatic_similarity']['same_output_between_5_correct']) / total_test_case_num,
                'same_output_between_5_timeout': len(problem_dic[name]['syntatic_similarity']['same_output_between_5_timeout']) / total_test_case_num,
                'same_output_between_5_exception': len(problem_dic[name]['syntatic_similarity']['same_output_between_5_exception']) / total_test_case_num,
                'same_output_between_5_execution_error': len(problem_dic[name]['syntatic_similarity']['same_output_between_5_execution_error']) / total_test_case_num,
                'Levenshtein_edit_distance': Levenshtein_edit_distance,
                'LED_all': LED_all
            }
        return syntatic_similarity_res

    def LCS(list1, list2):
        matcher = SequenceMatcher(None, list1, list2)
        match = matcher.find_longest_match(0, len(list1), 0, len(list2))
        return list1[match.a: match.a + match.size]

    def test_case_pass_rate_top5(code_candidates):
        # test_case_pass_rate = []
        tmp_test_case_pass_rate = []

        #  BLEU score with correct solution as reference
        tmp_list = []
        for code in code_candidates:
            if len(code['case_status'])!=0:
                tmp_test_case_pass_rate.append(float(len(code['passed_case']) / (len(code['case_status']))))
            else:
                tmp_test_case_pass_rate.append(float(0))
        # test_case_pass_rate.append(tmp_test_case_pass_rate)
        return tmp_test_case_pass_rate

    problem_dic = {}
    for seq in range(5):
        print('log/record/' + file_prefix + '.log_%s' % (seq))
        with open('log/record/' + file_prefix + '.log_%s' % (seq), 'r') as f:
        # with open('log/record/dataset_%s_model_%s_topn_5_temperature_%s.0.log_%s' % (dataset, model, temperature, seq), 'r') as f:
            for line in f.readlines():
                content = json.loads(line)
                name = content['name']
                if name not in problem_dic:
                    problem_dic[name] = {'code_candidates': []}
                index_num = content['index_num']
                code_candidates = content['code_candidates']
                code = code_candidates[0]
                problem_dic[name]['code_candidates'].append(code)

                if seq == 4:
                    code_candidates = []
                    code_reference = []
                    case_status_list = []
                    for code_res in problem_dic[name]['code_candidates']:
                        code_candidates.append(code_res['code'].split())
                        code_reference.append(code_res['code'])
                        case_status_list.append(code_res['case_status'])
                    # output equivalence rate
                    test_case_pass_rate = test_case_pass_rate_top5(problem_dic[name]['code_candidates'])
                    syntatic_similarity_res = syntatic_similarity(problem_dic, name, code_candidates, case_status_list)
                    problem_dic[name]['syntatic_similarity'] = syntatic_similarity_res
                    # LCS
                    LCS_list = []
                    LCS_all = []
                    for i in range(1, len(code_candidates)):
                        if len(code_candidates[0])!=0:
                            LCS_rate = len(LCS(code_candidates[0], code_candidates[i]))/len(code_candidates[0])
                        else:
                            LCS_rate = 0
                        LCS_list.append(LCS_rate)

                    for i in range(len(code_candidates)):
                        for j in range(len(code_candidates)):
                            if i != j:
                                if len(code_candidates[i]) != 0:
                                    LCS_rate = len(LCS(code_candidates[i], code_candidates[j])) / len(
                                        code_candidates[i])
                                else:
                                    LCS_rate = 0
                                LCS_all.append(LCS_rate)

                    problem_dic[name]['test_case_pass_rate'] = test_case_pass_rate
                    problem_dic[name]['LCS'] = LCS_list
                    problem_dic[name]['LCS_all'] = LCS_all
                    problem_dic[name].pop('code_candidates')
    json_str = json.dumps(problem_dic)
    with open(save_dir+'intermediate_result_top0_5.json', 'w') as f:
        f.write(json_str)

# def analyze_among_among5(dataset, model, temperature, seq):
def analyze_among_among5(file):
    dataset, model, temperature = '', '', ''
    file_str = file.split('/')[-1]
    file_suffix = file_str.split('.')[-1]
    file_prefix = file_str.replace('.'+file_suffix, '')
    file_prefix_str_list = file_prefix.split('_')
    type = file_prefix_str_list[0]
    for i, piece in enumerate(file_prefix_str_list):
        if piece == 'dataset':
            dataset = file_prefix_str_list[i+1]
        elif piece == 'model':
            model = file_prefix_str_list[i+1]
        elif piece == 'temperature':
            temperature = file_prefix_str_list[i+1]
    if 'topn_5' in file and 'latest' not in file:
        type = 'old'
    if dataset == 'code':
        dataset = 'code_contest'
    if type != 'dataset':
        save_dir = './syntactic_similarity/%s_%s_%s_%s/' % (type, dataset, model, temperature)
    else:
        save_dir = './syntactic_similarity/%s_%s_%s/' % (dataset, model, temperature)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save_dir = './result_data/%s_%s_%s/' % (dataset, model, temperature)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    def syntatic_similarity(problem_dic, name, code_candidates, case_status_list):
        same_output_between_5 = []
        same_output_between_5_correct = []
        same_output_between_5_timeout = []
        same_output_between_5_exception = []
        same_output_between_5_execution_error = []
        Levenshtein_edit_distance = []
        LED_all = []
        for i in range(len(case_status_list[0])):
            output_set = set()
            for j in range(len(problem_dic[name]['code_candidates'])):
                output_set.add(case_status_list[j][i])
            # print(output_set)
            if len(output_set) == 1:
                same_output_between_5.append(i)
                if list(output_set)[0] == 'timeout':
                    same_output_between_5_timeout.append(i)
                elif 'execution error' in list(output_set)[0]:
                    same_output_between_5_execution_error.append(i)
                elif list(output_set)[0] == 'exception' or sum([len(case) for case in code_candidates]) == 0:
                    same_output_between_5_exception.append(i)
                else:
                    same_output_between_5_correct.append(i)

        for i in range(len(problem_dic[name]['code_candidates'])):
            if i == 0:
                continue
            Levenshtein_edit_distance.append(nltk.edit_distance(code_candidates[0], code_candidates[i]))
        for i in range(len(problem_dic[name]['code_candidates'])):
            for j in range(i + 1, len(problem_dic[name]['code_candidates'])):
                LED_all.append(nltk.edit_distance(code_candidates[i], code_candidates[j]))
        problem_dic[name]['syntatic_similarity'] = {
            'same_output_between_5': same_output_between_5,
            'same_output_between_5_correct': same_output_between_5_correct,
            'same_output_between_5_timeout': same_output_between_5_timeout,
            'same_output_between_5_exception': same_output_between_5_exception,
            'same_output_between_5_execution_error': same_output_between_5_execution_error,
            'Levenshtein_edit_distance': Levenshtein_edit_distance,
            'LED_all': LED_all
        }
        total_test_case_num = len(problem_dic[name]['code_candidates'][0]['case_status'])
        if total_test_case_num == 0:
            syntatic_similarity_res = {
                'same_output_between_5': 0,
                'same_output_between_5_correct': 0,
                'same_output_between_5_timeout': 0,
                'same_output_between_5_exception': 0,
                'same_output_between_5_execution_error': 0,
                'Levenshtein_edit_distance': Levenshtein_edit_distance,
                'LED_all': LED_all
            }
        else:
            syntatic_similarity_res = {
                'same_output_between_5': len(problem_dic[name]['syntatic_similarity']['same_output_between_5']) / total_test_case_num,
                'same_output_between_5_correct': len(problem_dic[name]['syntatic_similarity']['same_output_between_5_correct']) / total_test_case_num,
                'same_output_between_5_timeout': len(problem_dic[name]['syntatic_similarity']['same_output_between_5_timeout']) / total_test_case_num,
                'same_output_between_5_exception': len(problem_dic[name]['syntatic_similarity']['same_output_between_5_exception']) / total_test_case_num,
                'same_output_between_5_execution_error': len(problem_dic[name]['syntatic_similarity']['same_output_between_5_execution_error']) / total_test_case_num,
                'Levenshtein_edit_distance': Levenshtein_edit_distance,
                'LED_all': LED_all
            }
        return syntatic_similarity_res

    def LCS(list1, list2):
        matcher = SequenceMatcher(None, list1, list2)
        match = matcher.find_longest_match(0, len(list1), 0, len(list2))
        return list1[match.a: match.a + match.size]

    def test_case_pass_rate_among5(code_candidates):
        # test_case_pass_rate = []
        tmp_test_case_pass_rate = []

        #  BLEU score with correct solution as reference
        tmp_list = []
        for code in code_candidates:
            if len(code['case_status'])!=0:
                tmp_test_case_pass_rate.append(float(len(code['passed_case']) / (len(code['case_status']))))
            else:
                tmp_test_case_pass_rate.append(float(0))
        # test_case_pass_rate.append(tmp_test_case_pass_rate)
        return tmp_test_case_pass_rate

    problem_dic = {}
    with open('log/record/dataset_%s_model_%s_topn_5_temperature_%s.log_%s' % (dataset, model, temperature, str(0)), 'r') as f:
        for line in f.readlines():
            content = json.loads(line)
            name = content['name']
            if name not in problem_dic:
                problem_dic[name] = {'code_candidates': []}
            index_num = content['index_num']
            code_candidates = content['code_candidates']
            # code = code_candidates[0]
            for code in code_candidates:
                problem_dic[name]['code_candidates'].append(code)

    for name in problem_dic:
        code_candidates = []
        code_reference = []
        case_status_list = []
        for code_res in problem_dic[name]['code_candidates']:
            code_candidates.append(code_res['code'].split())
            code_reference.append(code_res['code'])
            case_status_list.append(code_res['case_status'])
        test_case_pass_rate = test_case_pass_rate_among5(problem_dic[name]['code_candidates'])
        syntatic_similarity_res = syntatic_similarity(problem_dic, name, code_candidates, case_status_list)
        problem_dic[name]['syntatic_similarity'] = syntatic_similarity_res
        # LCS
        LCS_list = []
        LCS_all = []
        for i in range(1, len(code_candidates)):
            # BLEU score among 5 code candidates
            # the first one is important
            if len(code_candidates[0]) != 0:
                LCS_rate = len(LCS(code_candidates[0], code_candidates[i]))/len(code_candidates[0])
            else:
                LCS_rate = 0
            LCS_list.append(LCS_rate)

        for i in range(len(code_candidates)):
            for j in range(len(code_candidates)):
                if i != j:
                    if len(code_candidates[i]) != 0:
                        LCS_rate = len(LCS(code_candidates[i], code_candidates[j])) / len(
                            code_candidates[i])
                    else:
                        LCS_rate = 0
                    LCS_all.append(LCS_rate)
        problem_dic[name]['test_case_pass_rate'] = test_case_pass_rate
        problem_dic[name]['LCS'] = LCS_list
        problem_dic[name]['LCS_all'] = LCS_all
        problem_dic[name].pop('code_candidates')

    json_str = json.dumps(problem_dic)
    with open(save_dir+'intermediate_result_among5.json', 'w') as f:
        f.write(json_str)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-d",
    #     "--dataset",
    #     type=str,
    #     help="Choose dataset",
    #     required=True,
    # )
    # # 0, 1, 2
    # parser.add_argument(
    #     "-t",
    #     "--temperature",
    #     type=str,
    #     help="Choose temperature",
    #     required=True,
    # )
    # parser.add_argument(
    #     "-m",
    #     "--model",
    #     type=str,
    #     required=True,
    # )
    # parser.add_argument(
    #     "-o",
    #     "--option",
    #     type=str,
    #     choices=['R1', 'R2'],
    #     help="Choose the mode of the experiment",
    #     required=True,
    #     # default='original'
    # )
    # parser.add_argument(
    #     "-s",
    #     "--seq",
    #     type=int,
    #     # required=True,
    # )
    # parser.add_argument(
    #     "-f",
    #     "--file",
    #     type=str,
    #     help="Choose file",
    #     required=True,
    # )
    # args = parser.parse_args()
    # if args.option == 'R1':
    #     analyze_among_among5(args.dataset, args.model, args.temperature, args.seq)
    # elif args.option == 'R2':
        # analyze_among_top0_5(args.dataset, args.model, args.temperature)
    # analyze_among_top0_5(args.file)

    analyze_among_top0_5('log/record/CoT_dataset_HumanEval_model_gpt-3.5-turbo_topn_1_temperature_1.log_0')
    # analyze_among_among5('log/record/latest_dataset_APPS_model_gpt-3.5-turbo_topn_5_temperature_0.log_0')