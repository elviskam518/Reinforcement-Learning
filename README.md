# Learning to Walk with TD3

This reinforcement learning coursework project trains a simulated bipedal walker to move across **standard and obstacle-filled terrain**. I implemented a **Twin Delayed Deep Deterministic Policy Gradient (TD3)** agent in Python and PyTorch, then adapted its exploration and training configuration for the two environments.

The project explores a practical continuous-control problem: learning coordinated motor actions from experience while balancing progress, stability and failure penalties.

**[Read the project report](cqst66-agent-paper.pdf)** · **[Standard-terrain demonstration](cqst66-agent-video%2Cepisode%3D902%2Cscore%3D238.22761666221976.mp4)** · **[Hardcore demonstration](cqst66-agent-video-hardcore%2Cepisode%3D1897%2Cscore%3D227.80026134816924.mp4)**

## What I implemented

The agent uses an actor network to select continuous actions and two critic networks to estimate their value. The TD3 implementation includes:

- A replay buffer for learning from previously collected transitions.
- Twin critics, using the smaller target Q-value.
- Delayed actor updates and soft updates to target networks.
- Target-policy smoothing and exploration noise.
- Training logs, progress plots and recorded gameplay through the coursework's `rldurham` environment tools.

Both configurations use 400- and 300-unit hidden layers, a replay capacity of one million transitions, a batch size of 256 and a learning rate of 0.0003.

## Adapting the agent to different terrain

| Design choice | Standard terrain | Hardcore terrain |
| --- | --- | --- |
| Network stabilisation | LayerNorm and small output-layer initialisation | Omits these modifications |
| Initial exploration | No explicit random-action warmup | 25,000 random-action steps |
| Exploration noise | Reduced from 0.1 to 0.05 late in training | Gradual reduction from 0.1 to 0.01 |
| Additional adjustment | Late-training action scaling to 0.95 | Critic gradient clipping |
| Training length | 1,000 episodes | 2,000 episodes |

Both variants clip the reward used for learning to a minimum of −10. Recorded environment scores are tracked separately from this training reward.

## Results from the coursework report

| Observation | Standard terrain | Hardcore terrain |
| --- | ---: | ---: |
| Best recorded episode score | **238.2** | **227.8** |
| First positive-score episode | 166 | 318 |
| Episode where rolling-50 mean first exceeded 200 | 283 | 1,134 |

The report records only three negative-score episodes in the final 300 standard episodes, compared with 18 in the final 500 hardcore episodes.

These results describe the submitted training runs. Peak scores are individual episodes, not average evaluation performance across independent runs. The included videos illustrate selected episodes.

## Skills demonstrated

**Python · PyTorch · Gymnasium · rldurham · NumPy · Matplotlib**

The project demonstrates actor–critic implementation, continuous control, replay-based learning, exploration scheduling, training diagnostics and interpretation of learning curves.

## Limitations

The standard policy settles into a relatively conservative gait, while the hardcore policy still fails on some obstacles. The report discusses these trade-offs and proposes prioritised experience replay as future work; that extension is not implemented here.

The experiments do not isolate the contribution of each modification through a full ablation study or repeated-seed evaluation.

## Repository guide

- [`cqst66-agent-code.py`](cqst66-agent-code.py): standard-terrain TD3 agent and training loop.
- [`cqst66-agent-code-hardcore.py`](cqst66-agent-code-hardcore.py): hardcore variant.
- [`cqst66-agent-paper.pdf`](cqst66-agent-paper.pdf): methods, learning curves, results and limitations.
- [Standard log](cqst66-agent-log.txt) and [hardcore log](cqst66-agent-log-hardcore.txt): recorded training evidence.
- The two MP4 files provide demonstrations of the learned behaviours.
- [Assignment brief](Reinforcement_learning_coursework_description%281%29.pdf): task specification.

The scripts depend on the coursework-specific `rldurham` environment and begin training when run. The report also discloses AI assistance with training-log analysis and suggestions for future work.
