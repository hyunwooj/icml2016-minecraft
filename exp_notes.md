2018/01/21
==========

- Retention is used with attention at memory read.

- Using the existing reward for memory slot action is working but the diff and the avg retention reward is not working for some reasons.

- Commit: aeed726, tag: reten-same-reward

2018/01/17
==========

- Demonstrated that the agent can choose which memory slot to replace with the current frame by adding another output layer that approximates tthe action-value function for memory slot action.

- Exp: memory size 4, MQN, IMaze, success rate is 1.0

- Commit: d283aa9, tag: shared-q
