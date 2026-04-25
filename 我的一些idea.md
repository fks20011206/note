1.是不是要测一下非最终答案时doubt的热力图分布
2.
写一段代码， 将/data/DeepSeek-R1-Distill-Qwen-1.5B/data/output.jsonl 中的json内容中的提取出来，组成三类prompt,并且也以json的格式存储到/data/DeepSeek-R1-Distill-Qwen-1.5B/data/output2.jsonl

normal_prompt:
Solve the problem step by step.  
  
When you first reach a candidate answer, output:  
CANDIDATE_1: <answer>  
  
Then continue if needed and finally output:  
FINAL_ANSWER: <answer>  
  
Problem: ...

******************
self_doubt_prompt:
Solve the problem step by step.  
  
When you first reach a candidate answer, output:  
CANDIDATE_1: <answer>  
  
Then do 3 rounds of self-doubt and review.  
  
After round 1, output:  
CANDIDATE_2: <answer>  
  
After round 2, output:  
CANDIDATE_3: <answer>  
  
After round 3, output:  
CANDIDATE_4: <answer>  
  
At the end, output:  
FINAL_ANSWER: <answer>  
  
Problem: ...
*****************
direct_prompt:
Do not show reasoning. Output only:  
FINAL_ANSWER: <answer>  
  
Problem: ...
其中，存储的其他字段不变。

请帮我指定计划，我的模型代码文件在/home/ubuntu/exp-r1/code/transformers/src/transformers/models/qwen2先不要写代码，我想一开始只用normal_prompt进行训练，训练一个probe分类器，具体做法是：将candidate以及final_answer的位置都提取出来，并且提取对应token 的深层的hiddenstate，以如下表格的形式组织提取出的数据。
|question_id|traj_type|candidate_k|answer_text|final_answer|label|layer|vector|
|---|---|---|---|---|---|---|---|

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|gsm8k_1|self_doubt|1|684|694|0|20|h|

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|gsm8k_1|self_doubt|2|694|694|1|20|h|

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|gsm8k_1|self_doubt|3|694|694|1|20|h|

|   |   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|---|
|gsm8k_2|normal|1|17|17|1|20|h|

请评估后制定计划，并且告诉我还有什么地方需要补足