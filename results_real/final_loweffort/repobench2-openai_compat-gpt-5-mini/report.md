# Results: results_real/final_loweffort/repobench2-openai_compat-gpt-5-mini

60 runs.

## env=dynamic, repo=click

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.630 [0.491, 0.759] | 517,764 [251,931, 1,005,674] | 56.167 [42.667, 76.167] | 13.667 [11.333, 17.500] | 0.000 [0.000, 0.000] | 0.772 [0.650, 0.900] | 103.251 [60.219, 173.158] |

## env=dynamic, repo=flask

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.528 [0.417, 0.630] | 430,956 [313,128, 631,295] | 72.667 [64.667, 85.333] | 15.167 [14.000, 16.167] | 0.000 [0.000, 0.000] | 0.930 [0.870, 0.981] | 111.469 [85.657, 155.590] |

## env=dynamic, repo=httpx

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.667 [0.556, 0.750] | 374,230 [258,566, 544,094] | 82.000 [65.333, 104.000] | 20.000 [16.667, 23.333] | 0.000 [0.000, 0.000] | 0.976 [0.929, 1.000] | 127.791 [95.416, 182.716] |

## env=dynamic, repo=requests

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.574 [0.481, 0.657] | 162,539 [141,078, 183,566] | 52.833 [47.500, 60.000] | 13.000 [10.500, 15.500] | 0.000 [0.000, 0.000] | 0.869 [0.730, 0.976] | 58.648 [46.511, 70.838] |

## env=dynamic, repo=rich

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.731 [0.648, 0.806] | 449,175 [389,082, 494,482] | 104.333 [95.167, 112.167] | 30.000 [27.833, 31.833] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 142.104 [123.208, 164.795] |

## env=static, repo=click

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.898 [0.861, 0.944] | 434,661 [245,326, 791,824] | 54.667 [42.667, 73.167] | 13.500 [11.167, 17.333] | 0.000 [0.000, 0.000] | - | 85.107 [55.335, 140.809] |

## env=static, repo=flask

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.917 [0.843, 0.991] | 414,309 [304,045, 614,740] | 70.333 [62.167, 83.667] | 14.500 [13.667, 15.333] | 0.000 [0.000, 0.000] | - | 99.449 [80.447, 135.828] |

## env=static, repo=httpx

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.954 [0.907, 0.991] | 311,297 [248,768, 365,830] | 74.000 [63.000, 84.667] | 19.833 [16.833, 22.667] | 0.000 [0.000, 0.000] | - | 95.716 [82.963, 114.509] |

## env=static, repo=requests

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.963 [0.926, 0.991] | 160,087 [140,274, 178,357] | 50.833 [47.167, 54.500] | 11.667 [10.000, 13.667] | 0.000 [0.000, 0.000] | - | 52.961 [45.416, 59.798] |

## env=static, repo=rich

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| independent_evict | 6 | 0.935 [0.907, 0.963] | 443,594 [380,914, 481,431] | 103.667 [92.833, 112.333] | 29.667 [27.500, 31.667] | 0.000 [0.000, 0.000] | - | 129.973 [114.773, 146.215] |