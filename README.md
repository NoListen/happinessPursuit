# Modification
The code simplify the pong enviroment a lot and fix the asymmetry of score. 

The agents can play in both sides. The tuple action to the enviroment consists of up, down, idle or None. None means rule-based action. You can flip the image when the agent is playing in the right side so that it looks like playing in the left side. At the end, you can get one agent that can play well in both sides.

The DQN used in original work is not appreciated and needs further improvement.


# Happiness Pursuit
Code accompanying the paper "Happiness Pursuit: Personality Learning in a Society of Agents". TODO: add link to paper.

You can see the performance of the agents after training against hand-coded AI [here](https://youtu.be/tOVXdqVxCig).

To see the video showing the trained agents playing against each other click [here](https://youtu.be/LKtbF0ggvPQ).

# How to use
The repository contains the following main files:
* train.py - code for training the agents
* test.py - to test the trained agents
* match.py - to have 2 trained agents play against each other

See each file for details on how to run it.

# Main dependencies
* tensorflow
* cv2
* pygame

To install required dependencies, run `pip install -r requirements.txt`

Tested using Python 3.5.2
