# Results: results_real/final_gemini/repobench2-openai_compat-gemini-3.5-flash-lite

30 runs.

## env=dynamic, repo=click

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.500 | 146,323 | 37.000 | 9.000 | 0.000 | 0.000 | 9.795 |
| independent_evict | 1 | 0.500 | 196,685 | 23.000 | 9.000 | 0.000 | 0.200 | 16.138 |
| selective_llm | 1 | 0.500 | 136,901 | 25.000 | 9.000 | 0.000 | 0.400 | 9.795 |

## env=dynamic, repo=flask

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.611 | 189,062 | 65.000 | 12.000 | 0.000 | 0.000 | 31.361 |
| independent_evict | 1 | 0.444 | 268,815 | 30.000 | 14.000 | 0.000 | 1.000 | 20.723 |
| selective_llm | 1 | 0.611 | 188,250 | 63.000 | 12.000 | 0.000 | 0.000 | 31.361 |

## env=dynamic, repo=httpx

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.722 | 183,837 | 74.000 | 13.000 | 0.000 | 0.000 | 29.787 |
| independent_evict | 1 | 0.389 | 268,450 | 46.000 | 19.000 | 0.000 | 0.778 | 33.276 |
| selective_llm | 1 | 0.333 | 226,391 | 79.000 | 15.000 | 0.000 | 0.889 | 41.997 |

## env=dynamic, repo=requests

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.389 | 144,937 | 79.000 | 12.000 | 0.000 | 0.500 | 28.024 |
| independent_evict | 1 | 0.333 | 143,241 | 50.000 | 8.000 | 0.000 | 0.800 | 32.916 |
| selective_llm | 1 | 0.222 | 130,235 | 54.000 | 13.000 | 0.000 | 0.800 | 30.038 |

## env=dynamic, repo=rich

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.833 | 141,292 | 56.000 | 17.000 | 0.000 | 0.000 | 34.138 |
| independent_evict | 1 | 0.889 | 288,487 | 79.000 | 21.000 | 0.000 | 0.500 | 43.501 |
| selective_llm | 1 | 0.833 | 141,292 | 56.000 | 17.000 | 0.000 | 0.000 | 34.138 |

## env=static, repo=click

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.611 | 111,361 | 23.000 | 9.000 | 0.000 | - | 9.016 |
| independent_evict | 1 | 0.556 | 196,682 | 23.000 | 9.000 | 0.000 | - | 16.345 |
| selective_llm | 1 | 0.611 | 111,361 | 23.000 | 9.000 | 0.000 | - | 9.016 |

## env=static, repo=flask

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.611 | 140,017 | 47.000 | 10.000 | 0.000 | - | 22.410 |
| independent_evict | 1 | 0.833 | 268,815 | 30.000 | 14.000 | 0.000 | - | 20.723 |
| selective_llm | 1 | 0.611 | 140,017 | 47.000 | 10.000 | 0.000 | - | 22.410 |

## env=static, repo=httpx

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.778 | 227,879 | 69.000 | 16.000 | 0.000 | - | 40.563 |
| independent_evict | 1 | 0.778 | 268,450 | 46.000 | 19.000 | 0.000 | - | 33.276 |
| selective_llm | 1 | 0.778 | 227,879 | 69.000 | 16.000 | 0.000 | - | 40.563 |

## env=static, repo=requests

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.611 | 93,592 | 38.000 | 8.000 | 0.000 | - | 16.904 |
| independent_evict | 1 | 0.778 | 143,241 | 50.000 | 8.000 | 0.000 | - | 32.916 |
| selective_llm | 1 | 0.611 | 93,592 | 38.000 | 8.000 | 0.000 | - | 16.904 |

## env=static, repo=rich

| arch | n | question_accuracy | tokens_total | llm_calls | file_reads | kb_hits | stale_answer_rate | latency_sim_s |
|---|---|---|---|---|---|---|---|---|
| bacm_llm | 1 | 0.833 | 127,996 | 51.000 | 16.000 | 0.000 | - | 29.648 |
| independent_evict | 1 | 0.833 | 293,014 | 82.000 | 22.000 | 0.000 | - | 41.981 |
| selective_llm | 1 | 0.833 | 127,996 | 51.000 | 16.000 | 0.000 | - | 29.648 |