# Comparison table

| Method | Lessons | First attempt EM | 95% CI | First attempt F1 | Fixed | Broken | p | EM after 4 attempts | Added tokens | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| No memory | 0 | 65.0% | 59.4% to 70.2% | 0.779 |  |  |  | 66.0% | +0 | 0.30 |
| Reflexion (parent paper) | 0 | 64.3% | 58.8% to 69.5% | 0.776 | 0 | 2 | 0.500 | 72.7% | +5 | 0.28 |
| Lessons by failure type (proposed) | 3 | 66.7% | 61.2% to 71.8% | 0.802 | 16 | 11 | 0.442 | 75.0% | +189 | 0.45 |
| Random lessons | 3 | 67.0% | 61.5% to 72.1% | 0.796 | 15 | 9 | 0.307 | 75.7% | +193 | 0.35 |
| Lessons from similar questions | 3 | 69.7% | 64.2% to 74.6% | 0.819 | 23 | 9 | 0.020 | 79.7% | +177 | 0.42 |
| Lessons checked by retry only | 3 | 66.7% | 61.2% to 71.8% | 0.797 | 16 | 11 | 0.442 | 74.0% | +175 | 0.31 |
| Unchecked lessons | 3 | 66.0% | 60.5% to 71.1% | 0.786 | 15 | 12 | 0.701 | 75.7% | +186 | 0.33 |
| Lessons after a failed attempt | on retries | 65.3% | 59.8% to 70.5% | 0.782 | 1 | 0 | 1.000 | 72.0% | +4 | 0.30 |

Fixed and broken count first attempt answers that changed against the no memory
answer to the same question. p is the exact McNemar test on those two counts.
Added tokens and seconds are per question on the first attempt.
