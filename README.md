## Training

Training consists of 3 steps (given AP values for full validation dataset):
* Training from MobileNet weights. Expected AP after this step is ~38%.
* Training from weights, obtained from previous step. Expected AP after this step is ~39%.
* Training from weights, obtained from previous step and increased number of refinement stages to 3 in network. Expected AP after this step is ~40% (for the network with 1 refinement stage, two next are discarded).
