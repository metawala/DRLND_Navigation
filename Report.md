[//]: # (Image References)

[image1]: ./ReportImages/model.png "Model"
[image2]: ./ReportImages/Parameters.png "Agent Parameters"
[image3]: ./ReportImages/ScoresResults.png "Scores Results"

# Project Report : Navigation
This project report is in relation with the first project in DRLND course - Nativation. The task is to train an agent to collect **only** yellow bananas in the environment. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available.

## Learning Algorithm:

### About the paramteres:

Besides the model parameteres that are shown below in the image, our model uses an epsilon-greedy policy for selecting the next action. Following is how it is configured:
 - Epsilon start: 1.0
 - Epsilon end: 0.01
 - Epsilon decay: 0.995

In addition to this, we use a **replay buffer**. The buffer size is of 1000 with a minibatch sample of 64. This helps us avoid high correlation and have independent learning as explained in the lectures.
See the image below for all parameters defined in the code:

![Agent Parameters][image2]

For our Model, the class below is how it is structured.
Using a gamma of 0.99 and learning rate of 0.0005; we use RELU activation and simple nn architechture. Our FC units sizes are [128, 128]

![DQN Model][image1]

## Plots of Rewards:
The project is considered solved when the cumulative score reaches over 13. I ran into some issues when trying to get the score up. I continuously kept getting -ve scores after a first good run of 100 episodes. The issue was I was multiplying rewards with the `(gamma * QTargetsNext * (1 - dones))` instead of adding it to create a cumulative score.
I was able to achieve the target score in **775 episodes.** Below is an image that shows scores vs episode number.

![Scores Result][image3]

## Ideas for Future Work:
This project implements the RelayBuffer but there is room for trying out other novel methods that were part of the lecture series. 
1. For future work you could play around with:
    - [Prioritized Experienced Relay](https://arxiv.org/abs/1511.05952)
    - [Dueling Network Architectures](https://arxiv.org/abs/1511.06581)
    - [Double Q-learning](https://arxiv.org/abs/1509.06461)
2. I hardly played with the hyperparameters in this one, but I would want to experiment with the hyperparameteres to see how that affects learning and if I can improving the network.
3. We can train the network on direct images as well.