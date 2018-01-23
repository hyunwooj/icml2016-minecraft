2018/01/23
----------

- Memory initialization is crucial (Max value 31 failed, 127 succeeded).

- No-memory-reward (same) and diff-memory-reward (diff) work but avg-memory-reward (avg) does not.

- Need to verify diff-memory-reward affects learning and makes the performance better.

- Both ReLU and SoftPlus work but with memory size 4. Memory size 2 does not work.

- Not surprisingly, retention decreases exponentially. (Reten 1 at time 0, reten 0.37 at time 1 when stren 1.02)

- Tag: reten-diff-reward

| Model           | Succ. rate |
|-----------------|------------|
| same-softplus-3 | 0.0        |
| same-softplus-5 | 1.0        |
| same-relu-3     | 0.0        |
| same-relu-5     | 1.0        |
| diff-softplus-3 | 0.0        |
| diff-softplus-5 | 1.0        |
| diff-relu-3     | 0.033      |
| diff-relu-5     | 1.0        |
| avg-softplus-3  | 0.0        |
| avg-softplus-5  | 0.2        |


2018/01/21
----------

- Retention is used with attention at memory read.

- Using the existing reward for memory slot action is working but the diff and the avg retention reward is not working for some reasons.

- Tag: reten-same-reward

2018/01/17
----------

- Demonstrated that the agent can choose which memory slot to replace with the current frame by adding another output layer that approximates tthe action-value function for memory slot action.

- Exp: memory size 4, MQN, IMaze, success rate is 1.0

- Tag: shared-q
