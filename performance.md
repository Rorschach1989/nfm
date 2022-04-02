## Empirical evaluations

**instructions**: fill in corresponding evaluation metrics under the same random seed & replications  
with `mean(std)`

### Measured in c-index
|          | colon | whas | metabric | gbsg | flhain | support | kkbox |
|----------|-------|------|----------|------|--------|---------|-------|
| Ours     |       |      | 0.651    |      |        |         |       |
| DeepEH   |       |      |          |      |        |         |       |
| DeepHit  |       |      |          |      |        |         |       |
| Deepsurv |       |      |          |      |        |         |       |
| CoxTime  |       |      | 0。660    |      |        |         |       |


### Measured in integrated brier score
|          | colon | whas | metabric | gbsg | flhain | support | kkbox |
|----------|-------|------|----------|------|--------|---------|-------|
| Ours     |       |      | 0.178    |      |        |         |       |
| DeepEH   |       |      |          |      |        |         |       |
| DeepHit  |       |      |          |      |        |         |       |
| Deepsurv |       |      |          |      |        |         |       |
| CoxTime  |       |      | 0.166    |      |        |         |       |


