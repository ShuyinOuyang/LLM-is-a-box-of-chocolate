import json
import re
import numpy as np
import scipy.stats as stats
import os
import openpyxl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def ratio_of_worst(list, target):
    # calculating the ratio of worst cases happened in dataset
    # sepecially for test pass rate, OER, OER_ow
    count = 0
    for case in list:
        if case == target:
            count += 1

    return (count/len(list))

def semantic_syntactic_structural_similarity():
    if request_way == 'R1':
        with open(file_path + '/syntactic_similarity/%s%s_%s_%s/intermediate_result_among5.json' %
                  (option, dataset, model, temperature), 'r') as f:
            intermediate_result = json.load(f)
    else:
        with open(file_path + '/syntactic_similarity/%s%s_%s_%s/intermediate_result_top0_5.json' %
                  (option, dataset, model, temperature), 'r') as f:
            intermediate_result = json.load(f)

    test_case_pass_rate = []
    OER = []
    OER_ow = []
    LCS = []
    Levenshieten = []
    LCS_all = []
    LED_all = []

    # if request_way == 'R1':
    #     Levenshieten.append(intermediate_result['syntatic_similarity']['Levenshtein_edit_distance'])
    for case in intermediate_result:
        # if request_way == 'R1':
        #     OER.append(intermediate_result[case]['syntatic_similarity']['same_output_between_5'])
        #     OER_ow.append(intermediate_result[case]['syntatic_similarity']['same_output_between_5_correct'])
        # else:
        OER.append(intermediate_result[case]['syntatic_similarity']['same_output_between_5'])
        OER_ow.append(intermediate_result[case]['syntatic_similarity']['same_output_between_5_correct'])
        Levenshieten.append(intermediate_result[case]['syntatic_similarity']['Levenshtein_edit_distance'])
        LED_all.append(intermediate_result[case]['syntatic_similarity']['LED_all'])
        test_case_pass_rate.append(intermediate_result[case]['test_case_pass_rate'])
        LCS.append(intermediate_result[case]['LCS'])
        LCS_all.append(intermediate_result[case]['LCS_all'])

    # get structural similarity
    # if dataset == 'code_contest':
    #     if request_way == 'R1':
    #         with open(file_path + '/structural_similarity/%s_%s_structual_similarity_among5.json' % ('CodeContests', temperature), 'r') as f:
    #             problem_dic = json.load(f)
    #     else:
    #         with open(file_path + '/structural_similarity/%s_%s_structual_similarity_top0_5.json' % ('CodeContests', temperature), 'r') as f:
    #             problem_dic = json.load(f)
    # else:
    if request_way == 'R1':
        with open(file_path + '/structural_similarity/%s%s_%s_%s_structual_similarity_among5.json' %
                  (option, model, dataset, temperature), 'r') as f:
            problem_dic = json.load(f)
    else:
        with open(file_path + '/structural_similarity/%s%s_%s_%s_structual_similarity_top0_5.json' %
                  (option, model, dataset, temperature), 'r') as f:
            problem_dic = json.load(f)

    tmp = {
        'structual_similarity_UnifiedDiff': [],
        'structual_similarity_TreeDiff': [],
        'United_all': [],
        'Tree_all': []
    }
    for key in problem_dic:
        a = problem_dic[key]['structual_similarity']['structual_similarity_UnifiedDiff']
        # tmp['structual_similarity_UnifiedDiff'].append(a)
        if not isinstance(a[0], int):
            tmp['structual_similarity_UnifiedDiff'].append(a)
        else:
            tmp['structual_similarity_UnifiedDiff'].append([[0],[0],[0],[0]])
        a = problem_dic[key]['structual_similarity']['structual_similarity_TreeDiff']
        # tmp['structual_similarity_TreeDiff'].append(a)
        if not isinstance(a[0], int):
            tmp['structual_similarity_TreeDiff'].append(a)
        else:
            tmp['structual_similarity_TreeDiff'].append([[0],[0],[0],[0]])
        tmp_list = []
        for case in problem_dic[key]['structual_similarity']['United_all']:
            if not isinstance(case, int):
                tmp_list.append(case)
            else:
                tmp_list.append([0])
        tmp['United_all'].append(tmp_list)

        tmp_list = []
        for case in problem_dic[key]['structual_similarity']['Tree_all']:
            if not isinstance(case, int):
                tmp_list.append(case)
            else:
                tmp_list.append([0])
        tmp['Tree_all'].append(tmp_list)

        #     tmp['United_all'].append(problem_dic[key]['structual_similarity']['United_all'])
        # else:
        #     tmp['United_all'].append([[0] for _ in range(20)])
        # if not isinstance(problem_dic[key]['structual_similarity']['Tree_all'][0], int):
        #     tmp['Tree_all'].append(problem_dic[key]['structual_similarity']['Tree_all'])
        # else:
        #     tmp['Tree_all'].append([[0] for _ in range(20)])
        # # tmp['Tree_all'].append(problem_dic[key]['structual_similarity']['Tree_all'])

    United_Diff = tmp['structual_similarity_UnifiedDiff']
    Tree_Diff = tmp['structual_similarity_TreeDiff']
    United_all = tmp['United_all']
    Tree_all = tmp['Tree_all']

    res = {
        'test_case_pass_rate': test_case_pass_rate,
        'OER': OER,
        'OER_ow': OER_ow,
        'LED': Levenshieten,
        'LCS': LCS,
        'LED_all': LED_all,
        'LCS_all': LCS_all,
        'United_Diff': United_Diff,
        'Tree_Diff': Tree_Diff,
        'United_all': United_all,
        'Tree_all': Tree_all
    }

    return res
    # return test_case_pass_rate, OER, OER_ow, Levenshieten, LCS, LED_all, LCS_all, United_Diff, Tree_Diff, United_all, Tree_all

def get_correlation():
    # store all the fine-grained measurement in the dic named correlation (for later draw the heatmap)

    res = semantic_syntactic_structural_similarity()
    test_pass_rate = res['test_case_pass_rate']
    OER = res['OER']
    OER_ow = res['OER_ow']
    LED = res['LED']
    LCS = res['LCS']
    LED_all = res['LED_all']
    LCS_all = res['LCS_all']
    United_all = res['United_all']
    Tree_all = res['Tree_all']
    United_Diff = res['United_Diff']
    Tree_Diff = res['Tree_Diff']

    correlation = {'problem': [],
                   'test pass rate mean': [],
                   'test pass rate variance': [],
                   'test pass rate max diff': [],
                   'LCS mean': [],
                   'LCS min': [],
                   'LCS pair': [],
                   'LED mean': [],
                   'LED max': [],
                   'LED pair': [],
                   'United_Diff mean': [],
                   'United_Diff min': [],
                   'United_Diff pair': [],
                   'Tree_Diff mean': [],
                   'Tree_Diff min': [],
                   'Tree_Diff pair': [],
                   'description length': [],
                   'difficulty': [],
                   'time_limit': [],
                   'cf_rating': []
                   }

    test_pass_rate_var = [np.var(i) for i in test_pass_rate]
    test_pass_rate_var_avg = np.mean(test_pass_rate_var)
    test_pass_rate_max_diff = [max(i) - min(i) for i in test_pass_rate]
    test_pass_rate_max_diff_avg = np.mean(test_pass_rate_max_diff)

    for i in range(len(problem_list)):
        problem = problem_list[i]
        if dataset == 'HumanEval':
            correlation['problem'].append(problem['task_id'])
            correlation['description length'].append(len(problem['prompt']))

        elif dataset == 'APPS':
            correlation['problem'].append(problem['name'])
            correlation['description length'].append(len(problem['description']))
        else:
            correlation['problem'].append(problem['name'])
            correlation['description length'].append(len(problem['description']))
            correlation['difficulty'].append(problem['difficulty'])

            pattern = re.compile(r'(?<=seconds:=)*\d+')
            time_limit = pattern.findall(problem['time_limit'].split('\n')[0])[0]
            if 'seconds' in problem['time_limit']:
                correlation['time_limit'].append(int(time_limit))
            else:
                correlation['time_limit'].append(3)
            correlation['cf_rating'].append(problem['cf_rating'])

        correlation['test pass rate mean'].append(np.mean(test_pass_rate[i]))
        correlation['test pass rate variance'].append(np.var(test_pass_rate[i]))
        correlation['test pass rate max diff'].append(max(test_pass_rate[i])-min(test_pass_rate[i]))

    correlation['OER'] = OER
    correlation['OER_ow'] = OER_ow



    for case in LCS:
        correlation['LCS mean'].append(np.mean(case))
        # correlation['LCS variance'].append(np.var(case))
        correlation['LCS min'].append(min(case))
    for case in LCS_all:
        correlation['LCS pair'].append(np.mean(case))
    for case in LED:
        correlation['LED mean'].append(np.mean(case))
        # correlation['Levenshieten variance'].append(np.var(case))
        correlation['LED max'].append(max(case))
    for case in LED_all:
        correlation['LED pair'].append(np.mean(case))

    for case in United_Diff:
        correlation['United_Diff mean'].append(np.mean([i[0] for i in case]))
        # correlation['United_Diff variance'].append(np.var([i[0] for i in case]))
        correlation['United_Diff min'].append(min([i[0] for i in case]))
    for case in United_all:
        correlation['United_Diff pair'].append(np.mean([i[0] for i in case]))

    for case in Tree_Diff:
        correlation['Tree_Diff mean'].append(np.mean([i[0] for i in case]))
        # correlation['Tree_Diff variance'].append(np.var([i[0] for i in case]))
        correlation['Tree_Diff min'].append(min([i[0] for i in case]))
    for case in Tree_all:
        correlation['Tree_Diff pair'].append(np.mean([i[0] for i in case]))

    return correlation

def store_data_in_xlsx(correlation):
    # store in .xlsx
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    data = [[]]
    data[0].append(np.mean(correlation['test pass rate mean']))
    data[0].append(np.mean(correlation['test pass rate variance']))
    data[0].append(np.mean(correlation['test pass rate max diff']))
    data[0].append(ratio_of_worst(correlation['test pass rate max diff'], 1))
    data[0].append(np.mean(correlation['OER']))
    data[0].append(ratio_of_worst(correlation['OER'], 0))
    data[0].append(np.mean(correlation['OER_ow']))
    data[0].append(ratio_of_worst(correlation['OER_ow'], 0))
    data[0].append(np.mean(correlation['LCS mean']))
    data[0].append(np.mean(correlation['LCS min']))
    data[0].append(np.mean(correlation['LED mean']))
    data[0].append(np.mean(correlation['LED max']))
    data[0].append(np.mean(correlation['United_Diff mean']))
    data[0].append(np.mean(correlation['United_Diff min']))
    data[0].append(np.mean(correlation['Tree_Diff mean']))
    data[0].append(np.mean(correlation['Tree_Diff min']))

    for row in data:
        sheet.append(row)
    workbook.save('data.xlsx')

def draw_heatmap(correlation, save_dir):
    correlation_rank = []
    high_relavent = []
    problem_features = ['description length', 'difficulty', 'time_limit', 'cf_rating']
    for case in correlation_rank:
        if (case[0] in problem_features or case[1] in problem_features) and case[2][1] < 0.05:
            high_relavent.append(case)
            # print('%s & %s\'s correlation: %s' % (list(correlation.keys())[i],
            #                                       list(correlation.keys())[j],
            #                                       stats.pearsonr(correlation[list(correlation.keys())[i]], correlation[list(correlation.keys())[j]])
            #                                       )
            #       )
    correlation_list = []
    # test pass rate
    correlation_list.append(correlation['test pass rate mean'])
    correlation_list.append(correlation['test pass rate variance'])
    correlation_list.append(correlation['test pass rate max diff'])
    # output equivalence rate
    correlation_list.append(correlation['OER'])
    correlation_list.append(correlation['OER_ow'])
    # LCS
    correlation_list.append(correlation['LCS mean'])
    # correlation_list.append(correlation['LCS variance'])
    correlation_list.append(correlation['LCS min'])
    correlation_list.append(correlation['LCS pair'])
    # Levenshieten
    correlation_list.append(correlation['LED mean'])
    # correlation_list.append(correlation['Levenshieten variance'])
    correlation_list.append(correlation['LED max'])
    correlation_list.append(correlation['LED pair'])
    # United_Diff
    correlation_list.append(correlation['United_Diff mean'])
    # correlation_list.append(correlation['United_Diff variance'])
    correlation_list.append(correlation['United_Diff min'])
    correlation_list.append(correlation['United_Diff pair'])
    # Tree_Diff
    correlation_list.append(correlation['Tree_Diff mean'])
    # correlation_list.append(correlation['Tree_Diff variance'])
    correlation_list.append(correlation['Tree_Diff min'])
    correlation_list.append(correlation['Tree_Diff pair'])
    # problem features
    correlation_list.append(correlation['description length'])
    if dataset == 'code_contest':
        correlation_list.append(correlation['difficulty'])
        correlation_list.append(correlation['time_limit'])
        correlation_list.append(correlation['cf_rating'])

    if dataset == 'code_contest':
        column_names = ['TPR mean value',
                        'TPR mean variance',
                        'TPR mean max diff',

                        'OER mean',
                        'OER (no ex.) mean',

                        'LCS mean',
                        'LCS worst',
                        'LCS pair mean',
                        'LED mean',
                        'LED worst',
                        'LED pair mean',
                        'United_Diff mean',
                        'United_Diff worst',
                        'United_Diff pair mean',
                        'Tree_Diff mean',
                        'Tree_Diff worst',
                        'Tree_Diff pair mean',
                        'description length',
                        'difficulty',
                        'time_limit',
                        'cf_rating'
                        ]
    else:
        column_names = ['TPR mean value',
                        'TPR mean variance',
                        'TPR mean max diff',

                        'OER mean',
                        'OER_ow mean',

                        'LCS mean',
                        'LCS worst',
                        'LCS pair mean',
                        'LED mean',
                        'LED worst',
                        'LED pair mean',
                        'United_Diff mean',
                        'United_Diff worst',
                        'United_Diff pair mean',
                        'Tree_Diff mean',
                        'Tree_Diff worst',
                        'Tree_Diff pair mean',

                        'description length'
                        ]

    p_values = []
    correlation_values = []
    empty_values = []
    for i in range(len(column_names)):
        p_tmp = []
        c_tmp = []
        e_tmp = []
        for j in range(len(column_names)):
            p_tmp.append(stats.pearsonr(correlation_list[i], correlation_list[j])[1])
            c_tmp.append(stats.pearsonr(correlation_list[i], correlation_list[j])[0])
            e_tmp.append(0)
        p_values.append(p_tmp)
        correlation_values.append(c_tmp)
        empty_values.append(e_tmp)

    for i in range(len(column_names)):
        for j in range(len(column_names)):
            if p_values[i][j] > 0.05:
                empty_values[i][j] = '-'
            else:
                empty_values[i][j] = round(correlation_values[i][j], 2)


    fig, ax = plt.subplots(figsize=(20, 20))
    fig.subplots_adjust(top=0.98, bottom=0.2, left=0.2)
    p1 = sns.heatmap(correlation_values, annot=empty_values, cmap='Greys',
                     xticklabels=column_names, yticklabels=column_names, annot_kws={"fontsize": 16}, fmt='')

    cbar = p1.collections[0].colorbar
    # Set the font size of the color bar labels
    cbar.ax.tick_params(labelsize=20)
    # p1.yticks(rotation=90)
    p1.set_xticklabels(p1.get_xticklabels(), fontsize=25)
    p1.tick_params(axis='y', labelsize=25)

    # plt.show()
    plt.savefig(save_dir + 'heatmap_metric_%s_%s.pdf' % (dataset, temperature))

def confidence_interval(data, alpha=0.05, print=True):
    sample_mean = np.mean(data)
    sample_variance = np.var(data)
    sample_size = len(data)

    sample_std = sample_variance ** 0.5

    t_critical = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)

    margin_of_error = t_critical * (sample_std / (sample_size ** 0.5))
    # lower_bound = sample_mean - margin_of_error
    # upper_bound = sample_mean + margin_of_error
    if print:
        return '%s$\pm$%s'% (round(sample_mean,2), round(margin_of_error,2))
    else:
        return sample_mean, margin_of_error

def draw_box_plot_conversation(test_pass_rate):
    fontsize=26
    tmp_list = []
    for i in range(5):
        tmp_test_pass_rate = [case[i] for case in test_pass_rate]
        tmp_list.append(tmp_test_pass_rate)


    means = [np.mean(d) for d in tmp_list]
    conf_intervals = [confidence_interval(d, print=False)[1] for d in tmp_list]
    print(means, conf_intervals)
    plt.figure(figsize=(8, 8))
    x_values = range(1, 6)

    plt.plot(x_values, means, marker='o', linestyle='-', color='b')

    plt.errorbar(x_values, means, yerr=conf_intervals, fmt='o', ecolor='g', capsize=5, capthick=2)
    plt.xlabel('Number of requests', fontsize=fontsize)
    plt.ylabel('Mean test pass rate', fontsize=fontsize)
    plt.ylim(0.09, 0.48)
    # axs[index].set_ylim(bottom=0)
    # plt.set_title(dataset, fontsize=18)
    plt.xticks(x_values, ['1', '2', '3', '4', '5'], fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.subplots_adjust(left=0.22, bottom=0.2)
    # plt.tight_layout()
    plt.savefig('./major_revision/conversation_line_char_%s.pdf' % dataset)
    plt.show()

def print_all_measurement_summary():
    res = semantic_syntactic_structural_similarity()

    test_pass_rate = res['test_case_pass_rate']
    test_pass_rate_var = [np.var(x) for x in test_pass_rate]
    test_pass_rate_mean = [np.mean(x) for x in test_pass_rate]
    test_pass_rate_max_diff = [max(x)-min(x) for x in test_pass_rate]
    # test_pass_rate_ratio_of_worst = ratio_of_worst(test_pass_rate_max_diff, 1)
    print('test_pass_rate')
    print('mean value', 'mean variance', 'mean max diff', 'max diff', 'ratio of worst cases')
          # 'test_pass_rate_max_diff', 'test_pass_rate_ratio_of_worst')
    print('%s & %s & %s & %s & %s' % (confidence_interval(test_pass_rate_mean),
                                      confidence_interval(test_pass_rate_var),
                                      confidence_interval(test_pass_rate_max_diff),
                                      '1.00',
                                      str(round(ratio_of_worst(test_pass_rate_max_diff, 1) * 100, 2)) + '\%',
                                      # ratio_of_worst(test_pass_rate_max_diff, 1)
                                      ))
    # print('test_pass_rate_mean', confidence_interval(test_pass_rate_mean))
    # print('test_pass_rate_var', confidence_interval(test_pass_rate_var))
    # print('test_pass_rate_max_diff', confidence_interval(test_pass_rate_max_diff))
    # print('test_pass_rate_ratio_of_worst', test_pass_rate_ratio_of_worst)

    # Test pass rate per conversation

    if option == 'conversation_':
        for i in range(5):
            tmp_test_pass_rate = [case[i] for case in test_pass_rate]
            print('%s, mean test pass rate: %s' % (i, confidence_interval(tmp_test_pass_rate)))


    print('OER, OER_ow')
    print('mean value', 'worst value', 'ratio of worst cases')
    print('%s & %s & %s & %s & %s & %s' % (confidence_interval(res['OER']),
                                           '0.00',
                                           str(round(ratio_of_worst(res['OER'], 0)*100, 2)) + '\%',
                                           confidence_interval(res['OER_ow']),
                                           '0.00',
                                           str(round(ratio_of_worst(res['OER_ow'], 0)*100, 2)) + '\%'
                                           ))

    # print('OER', confidence_interval(res['OER']))
    # print('OER_ow', confidence_interval(res['OER_ow']))

    LCS_mean = [np.mean(x) for x in res['LCS']]
    LCS_min = [min(x) for x in res['LCS']]
    LCS_mean_in_pairs = [np.mean(x) for x in res['LCS_all']]
    LED_mean = [np.mean(x) for x in res['LED']]
    LED_max = [max(x) for x in res['LED']]
    LED_mean_in_pairs = [np.mean(x) for x in res['LED_all']]

    # print('LCS_mean', confidence_interval(LCS_mean))
    # print('LCS_min', confidence_interval(LCS_min))
    # print('LCS_mean_in_pairs', confidence_interval(LCS_mean_in_pairs))
    #
    # print('LED_mean', confidence_interval(LED_mean))
    # print('LED_max', confidence_interval(LED_max))
    # print('LED_mean_in_pairs', confidence_interval(LED_mean_in_pairs))

    print('LCS, LED')
    print('mean value', 'mean worst value', 'pair mean value')
    print('%s & %s & %s & %s & %s & %s' % (confidence_interval(LCS_mean),
                                           confidence_interval(LCS_min),
                                           confidence_interval(LCS_mean_in_pairs),
                                           confidence_interval(LED_mean),
                                           confidence_interval(LED_max),
                                           confidence_interval(LED_mean_in_pairs)
                                           ))


    United_Diff_mean = []
    United_Diff_min = []
    United_Diff_in_pairs = []
    for case in res['United_Diff']:
        # if case != [[0],[0],[0],[0]]:
        United_Diff_mean.append(np.mean([i[0] for i in case]))
        United_Diff_min.append(min([i[0] for i in case]))

    for index, case in enumerate(res['United_all']):
        United_Diff_in_pairs.append(np.mean([i[0] for i in case]))

    Tree_Diff_mean = []
    Tree_Diff_min = []
    Tree_Diff_in_pairs = []
    for case in res['Tree_Diff']:
        # if case != [[0],[0],[0],[0]]:
        Tree_Diff_mean.append(np.mean([i[0] for i in case]))
        Tree_Diff_min.append(min([i[0] for i in case]))
    for case in res['Tree_all']:
        Tree_Diff_in_pairs.append(np.mean([i[0] for i in case]))
    # print('United_Diff_mean', confidence_interval(United_Diff_mean))
    # print('United_Diff_min', confidence_interval(United_Diff_min))
    # print('United_Diff_in_pairs', confidence_interval(United_Diff_in_pairs))
    # print('Tree_Diff_mean', confidence_interval(Tree_Diff_mean))
    # print('Tree_Diff_min', confidence_interval(Tree_Diff_min))
    # print('Tree_Diff_in_pairs', confidence_interval(Tree_Diff_in_pairs))

    print('United_Diff, Tree_Diff')
    print('mean value', 'mean worst value', 'pair mean value')
    print('%s & %s & %s & %s & %s & %s' % (confidence_interval(United_Diff_mean),
                                           confidence_interval(United_Diff_min),
                                           confidence_interval(United_Diff_in_pairs),
                                           confidence_interval(Tree_Diff_mean),
                                           confidence_interval(Tree_Diff_min),
                                           confidence_interval(Tree_Diff_in_pairs)
                                           ))



if __name__ == "__main__":
    # config (change to apply)
    dataset_ = ['code_contest', 'APPS', 'HumanEval']
    dataset = dataset_[2]

    request_way_ = ['R1', 'R2']
    request_way = request_way_[1]
    temperature_ = [0, 1, 2]
    temperature = temperature_[0]
    problem_list = []
    # gpt-3.5-turbo or gpt-4
    model_ = ['gpt-3.5-turbo', 'gpt-4']
    model = model_[0]
    option = ''

    # customized
    file_path = ''

    print('dataset:%s\nrequest_way:%s\ntemperature:%s\nmodel:%s\noption:%s\n' % (dataset, request_way, temperature, model, option))

    if dataset == 'code_contest':
        # with open('./tmp2/code_contests_test.json', 'r') as f:
        with open('./dataset/code_contests_test.json', 'r') as f:
            problem_list = json.load(f)
    elif dataset == 'HumanEval':
        with open('./HumanEval/HumanEval.jsonl', 'r') as f:
            for line in f.readlines():
                problem_list.append(json.loads(line))
    elif dataset == 'APPS':
        path = './APPS/test/'
        for dirpath, dirnames, filenames in os.walk(path):
            # iterating for every problem
            for dirname in dirnames[:500]:
                # description
                with open(path + dirname + '/question.txt', 'r', encoding='utf-8') as f:
                    description = f.read()
                problem_list.append({'name': dirname, 'description': description})

    res = semantic_syntactic_structural_similarity()
    print_all_measurement_summary()

    # correlation = get_correlation()
    # draw_heatmap(correlation, 'major_revision/fig/')