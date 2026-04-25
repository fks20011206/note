1.是不是要测一下非最终答案时doubt的热力图分布
2.
将/data/DeepSeek-R1-Distill-Qwen-1.5B/data/output.jsonl 中的json内容中的提取出来，组成三类prompt,并且也以json的格式存储到/data/DeepSeek-R1-Distill-Qwen-1.5B/data/output2.jsonl

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
其中，存储的子段位