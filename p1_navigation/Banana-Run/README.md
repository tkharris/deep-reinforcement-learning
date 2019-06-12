# Banana Run

This is my solution to the bananas problem for the Udacity Deep Reinforcement Learning Nanodegree (Summer Session 2019) Udacity adapted this problem and runtime environment from the Unity's ML-Agents Banana Collector environment. This is the first of 5 projects.

The purpose of project is educational; to reinforce and deepen my understanding of value-based methods for Deep Reinforcement Learning (DRL). I will attempt to demonstrate a useful skill, in my ability to quickly construct a good solution to a problem which is a stand-in for a large class of real-world problems. I will also attempt to develop an academic comprehension of DNN value-based methods for reinforcement learning by analysing their performance characteristics and comparing them to other forms of machine learning. It is that kind of understanding that may lead me to invent and evaluate novel solutions to similar problems.

## Setting up

Assuming that you have an Nvidia GPU, you can get the Nvidia Pytorch Docker Container and use that for training etc.

Install dependencies...

```fish
git clone git@github.com:tkharris/deep-reinforcement-learning.git
docker pull tk/nvidia-tk:dog-project
```
_Should I post this container somehow or find a way to get this to work with a vanilla nvcr.io/nvidia/pytorch:19.05-py3?_

Run the container...

```fish
cd deep-reinforcement-learning/p1_navigation/Banana-Run
nvidia-docker run -it --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v (pwd):/workspace/Banana-Run tk/nvidia-tk:dog-project
```

And once in the docker container...

```bash
cd Banana-Run
conda activate dog-project
jupyter-notebook --port 6006 --ip 0.0.0.0 --allow-root Banana-Run.ipynb
```

## Methods, Techniques, and Evaluations

Details on the methods, techniques, and evaluations can be found in the Notebook proper.
