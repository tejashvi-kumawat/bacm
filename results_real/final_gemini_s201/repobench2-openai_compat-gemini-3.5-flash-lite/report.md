# Results: results_real/final_gemini_s201/repobench2-openai_compat-gemini-3.5-flash-lite

30 runs.

## env=dynamic, repo=click

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.389 | 267,605 | 85.000 | 11.000 | 0.000 | 0.000 | 39.134 |
| independent_evict | 1 | 0.444 | 227,432 | 33.000 | 9.000 | 0.000 | 0.167 | 30.092 |
| selective_llm | 1 | 0.389 | 258,302 | 72.000 | 11.000 | 0.000 | 0.333 | 39.019 |

## env=dynamic, repo=flask

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.278 | 231,676 | 66.000 | 11.000 | 0.000 | 0.333 | 39.265 |
| independent_evict | 1 | 0.278 | 325,136 | 55.000 | 12.000 | 0.000 | 1.000 | 38.455 |
| selective_llm | 1 | 0.111 | 181,245 | 59.000 | 9.000 | 0.000 | 0.444 | 41.154 |

## env=dynamic, repo=httpx

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.556 | 110,479 | 60.000 | 7.000 | 0.000 | 0.000 | 29.071 |
| independent_evict | 1 | 0.444 | 192,364 | 62.000 | 8.000 | 0.000 | 0.500 | 84.531 |
| selective_llm | 1 | 0.556 | 110,479 | 60.000 | 7.000 | 0.000 | 0.000 | 29.071 |

## env=dynamic, repo=requests

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.722 | 99,682 | 49.000 | 15.000 | 0.000 | 0.000 | 28.085 |
| independent_evict | 1 | 0.556 | 124,475 | 37.000 | 11.000 | 0.000 | 0.143 | 39.196 |
| selective_llm | 1 | 0.611 | 89,997 | 39.000 | 15.000 | 0.000 | 0.143 | 30.990 |

## env=dynamic, repo=rich

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.667 | 109,617 | 44.000 | 15.000 | 0.000 | 0.000 | 23.842 |
| independent_evict | 1 | 0.222 | 349,550 | 85.000 | 14.000 | 0.000 | 0.200 | 94.092 |
| selective_llm | 1 | 0.667 | 109,617 | 44.000 | 15.000 | 0.000 | 0.000 | 23.842 |

## env=static, repo=click

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.500 | 171,324 | 54.000 | 9.000 | 0.000 | - | 27.841 |
| independent_evict | 1 | 0.500 | 222,869 | 34.000 | 9.000 | 0.000 | - | 31.267 |
| selective_llm | 1 | 0.500 | 171,324 | 54.000 | 9.000 | 0.000 | - | 27.841 |

## env=static, repo=flask

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.444 | 145,394 | 46.000 | 7.000 | 0.000 | - | 23.182 |
| independent_evict | 1 | 0.778 | 325,136 | 55.000 | 12.000 | 0.000 | - | 38.455 |
| selective_llm | 1 | 0.444 | 145,394 | 46.000 | 7.000 | 0.000 | - | 23.182 |

## env=static, repo=httpx

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.556 | 116,987 | 61.000 | 9.000 | 0.000 | - | 31.213 |
| independent_evict | 1 | 0.556 | 107,691 | 42.000 | 7.000 | 0.000 | - | 56.432 |
| selective_llm | 1 | 0.556 | 116,987 | 61.000 | 9.000 | 0.000 | - | 31.213 |

## env=static, repo=requests

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.667 | 51,388 | 27.000 | 7.000 | 0.000 | - | 23.117 |
| independent_evict | 1 | 0.611 | 100,655 | 32.000 | 9.000 | 0.000 | - | 38.526 |
| selective_llm | 1 | 0.667 | 51,388 | 27.000 | 7.000 | 0.000 | - | 23.117 |

## env=static, repo=rich

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.722 | 94,116 | 38.000 | 15.000 | 0.000 | - | 22.066 |
| independent_evict | 1 | 0.278 | 349,550 | 85.000 | 14.000 | 0.000 | - | 93.852 |
| selective_llm | 1 | 0.722 | 94,116 | 38.000 | 15.000 | 0.000 | - | 22.066 |