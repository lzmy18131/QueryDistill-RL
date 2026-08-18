# BIRD Data Report (Pilot Subset)

- Generated: {"source": "birdsql/bird23-train-filtered + bird_mini_dev"}
- Train count: 50
- Dev count: 20
- Test count: 0
- Database count: 3
- Databases: book_publishing_company, car_retails, debit_card_specializing
- Gold execution success: 70/70

## Token statistics

| Metric | P50 | P90 | P95 | P99 | Max |
| --- | --- | --- | --- | --- | --- |
| Schema tokens | 751 | 751 | 751 | 751 | 751 |
| Evidence tokens | 18 | 59 | 65 | 69 | 107 |
| Raw prompt tokens | 814 | 838 | 850 | 866 | 888 |
| Chat-serialized prompt tokens | 822 | 846 | 858 | 874 | 896 |

## Truncation rate by max_prompt_length

| max_prompt_length | examples over budget | rate |
| --- | --- | --- |
| 512 | 50 | 71.43% |
| 768 | 44 | 62.86% |
| 1024 | 0 | 0.00% |
| 1536 | 0 | 0.00% |

## Note

This report is generated from the real BIRD pilot subset; it is not a benchmark claim.
