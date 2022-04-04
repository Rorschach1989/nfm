## Empirical evaluations

**instructions**: fill in corresponding evaluation metrics under the same random seed & replications  
with `mean(std)`

### Measured in c-index
|          | colon        | whas         | metabric     | gbsg         | flchain      | support      | kkbox        |
|----------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Ours     |              |              | 0.651        |              |              |              |              |
| DeepEH   |              |              |              |              |              |              |              |
| DeepHit  |              |              |              |              |              |              |              |
| Deepsurv |              |              |              |              |              |              |              |
| CoxTime  | 0.645(0.032) | 0.783(0.023) | 0.660(0.020) | 0.672(0.017) | 0.790(0.010) | 0.614(0.010) |              |


### Measured in integrated brier score
|          | colon        | whas         | metabric     | gbsg         | flchain      | support      | kkbox        |
|----------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| Ours     |              |              | 0.178        |              |              |              |              |
| DeepEH   |              |              |              |              |              |              |              |
| DeepHit  |              |              |              |              |              |              |              |
| Deepsurv |              |              |              |              |              |              |              |
| CoxTime  | 0.189(0.012) | 0.135(0.011) | 0.166(0.008) | 0.179(0.006) | 0.103(0.007) | 0.192(0.004) |              |


