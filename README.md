# Learning how to play tennis using DRL

This repository provides code to for ML agents to learn how to play tennis using deep reinforcement learning in the environment provided by the [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

## Introduction

![Trained Agent](./assets/tennis.gif)

In the tennis enviroment, two players control their respective racket to play the ball over the net. The goal of the enviroment is to train a ML agent to play the ball over the net, without playing it out of bounds - thus learning to play a decent game of tennis. 

*(Note that the goal of this environment is not to play a "correct" game of tennis following all the rules - it's more comparable to a child learning the basic movements needed to hit a ball with a racket in a controlled manner).*

## Learning Task

The learning task is given by the following table

Attribute | Description
--- | ---
Goal | Keep the tennis ball in play.
Goal (RL) | Maximize expected cumulative reward.
Observations | `Box(8)`, eight continuous variables correspond to the position and velocity of the ball and racket. Each agent receives its own local observation. The actual observation is a stacked vector of three observations, thus resulting in a `Box(24)` continuous vector observation.
Actions | `Box(2)`, two continuous variables correspond to the movement in direction of the net (positive or negative) and to a jumping movement.
Rewards | +0.1 for hitting the ball over the net <br> -0.01 for hitting the ball out of bounds

The task is episodic and ends whenever the ball touches the ground. In order to solve the environment, an **average score of +0.5** over 100 consecutive episodes is needed. The score is computed as follows:
- After each episode, the rewards for each agent are added up (no discounting), to get a score for each agent. 
- The maximum of these two (potentially different) scores is taken.
- This yields a single **score** for each episode.

## Installation

This project is using Python 3.6.3, make sure the following packages are installed:

```bash
pip install numpy matplotlib torch setuptools bleach==1.5.0 unityagents
```

Download the environment from one of the links below:
- Linux: [download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
- Headless (Linux): [download](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)

Place the downloaded file(s) in the `src/exec/` folder, and unzip (or decompress) the file. Make sure to update the `file_name` parameter in the code when loading the environment:

```python
env = UnityEnvironment(file_name="src/exec/...")
```

## Training instructions

Follow the instructions in `training.ipynb` to see how an agent can be trained.

Coming soon...

## Experiments

Coming soon...

## Further Work

Coming soon...
