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



我现在的方法是：

先得到 normal trajectory
在 candidate 处分支
一条让模型正常收尾
一条让模型续写 self-doubt 段（续写self-doubt段采用prompt引导的方式，多次采样，并且与分支前的文本进行拼接）：
再对完整文本 forward。

probe除了训练对答案的置信度后，还需要训练未来是否会继续self-doubt。
训练完probe后，组织一个json文件，并提取以下信息，question_id、traj_type、candidate（包含candidate_pos,candidate_ans,candidate_ans是经过规则处理的也就是只提取了数字的）、answer_text（输出的所有内容）、final_answer（若不存在，就选择最后一个candidate）、


看后续 candidate / final answer 的 hidden state，经 probe 后，置信度是否上升。