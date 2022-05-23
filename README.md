# Udacity Deep Reinforcement Learning course - Project 1: Continuous control


This repository contains my solution for the second project of Udacity's course on Reinforcement Learning. The scenario's goal is to control a double-jointed arm keeping the tip on a moving target.
## Contents
This repo contains:

## Getting Started

This project was developed and tested on an Apple Macbook with an intel i5. No guarantees are made on performance in other systems. 

### Mac

To setup your coding evironment you need to perform 3 steps after cloning this repository:

1. Make `setup.sh` executable. There are many ways to do this. One way is through the terminal run this command:

```bash
chmod +x setup.py
```

2. Then you simply run `setup.py`.

3. Finally you activate the conda environment in your terminal or on your notebook change the kernel to `drl_navigation`
### Others

If you are running this on other operating systems. There's a strong possibility that you can just follow the instructions for mac. Otherwise you will need to follow the steps in the [readme](Value-based-methods/README.md) in the `Value-based-methods` repo.

#### To download the environment for other Operating Systems
If you are using anothe OS you'll need to manually download the environment from one of the links below.  You need only select the environment that matches your operating system:

Version 1: One (1) Agent
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

Version 2: Twenty (20) Agents
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


## The Environment

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.