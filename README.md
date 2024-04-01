# LLM is Like a Box of Chocolates: An Empirical Study of the Non-determinism of ChatGPT in 0-shot Code Generation

## Datasets
There are links to the datasets we used in our experiments. (Below are the instructions on how to download and prepare the dataset)

[HumanEval](https://github.com/openai/human-eval) 

1. Click the link will redirect you to the human-eval homepage. On the homepage, you can download the data from '.\data\HumanEval.jsonl.gz'.
2. Decompress the file, then you get the dataset ''.\data\HumanEval.jsonl'
3. Due to the dataset are not perfect for the experiment, we use script [Modify_HumanEval.py](https://github.com/CodeHero0/Stability-of-ChatGPT-in-Code-Generation/blob/main/Modify_HumanEval.py) to reshape the HumanEval dataset.

[APPS](https://github.com/hendrycks/apps)
1. Click the link will redirect you to the apps homepage.
2. Dataset could be downloaded from the link shown in README.md 'Download the APPS dataset here.(~1.3GB)'
3. Decompress the file, then you get the dataset


[CodeContests](https://github.com/deepmind/code_contests) 
1. Click the link will redirect you to the code_contests homepage.
2. Follow the instruction shown in README.md, your could download the dataset with .riegeli file.
3. For further easy use, in our experiment, we use the [script](https://github.com/deepmind/code_contests/pull/21) to convert riegeli files into JSON files.


## Experiments
Our experiments mainly contain the following scripts:

### 1. Generating response

We use generate_response.py to generate responses with three code problem datasets, and store them into JSON files in '.\log\'. (Below is the example command to run this script). If you want to change the way of requesting ChatGPT, please see [Openai's official website](https://platform.openai.com/docs/api-reference/chat).


```sh
python generate_response.py
```

There are bugs that might occur in running this script:
1. Dataset path error (change the dataset store path in the functions)
2. Config your own openai info in file `openai_info/info.json`. (**DON'T upload your script with your openai.key to the public repository!!!!!!!!!!!**) If you are still a free user of Openai model API, please config your account as pay-as-you-go before running the script, otherwise it's likely to reach the free trial limitation.
3. Download the library that you don't have in your environment
4. Due to some unexpected error from ChatGPT API such as requesting over time, running the script once cannot guarantee to cover all the problems in the dataset. So our recommended solution is to run the script multiple times.


### 2. Intermediate result processing

We use `intermedia_analyze.py` to generate intermedia based on the test cases and responses we get in the first step, intermedia results will be stored in `'.\log\record\'`. (Below is the example command to run this script)

```sh
python intermedia_analyze.py -f log/dataset_APPS_model_gpt-3.5-turbo_topn_5_temperature_0.0.log_0
```
There are bugs that might occur in running this script:
1. The parameter after -f should start with `'log/'`
2. Dataset path error (change the dataset store path in the functions, e.g. `code_contest_analyze_process`, `analyze_process_HumanEval`, and `analyze_process_APPS`)


We then use `syntactic_similarity_OER.py` and `structural_similarity.py` to generate the final results in semantic similarity, syntactic similarity, and structural similarity. `syntactic_similarity_OER.py` is for semantic similarity and syntactic similarity measurements, e.g. test pass rate, OER, OER_ow, LCS, LED. `structural_similarity.py` is for structural similarity measurements, e.g. United_Diff, Tree_Diff. (Below is the example command to run this script)
```sh
python syntactic_similarity_OER.py
```
```sh
python structural_similarity.py
```
There are bugs that might occur in running this script:
1. The config settings are written in each code's main function, change the config setting before running the script
2. For `structural_similarity.py`, the 'Tree_Diff' measurement needs the library `zss`. If not downloaded, please use `pip install zss`. For more detail about the library `pycode_similar`, please see its [GitHub page](https://github.com/fyrestone/pycode_similar).


### 3. Result display

We use `measurement_summary_draw_heatmap.py` to display our result. (Please modify the config setting before running the script)

## Full results

'./images' stores all the experiment images used in the paper.

'./log' stores part of the experiment logs.
