# -*- coding: utf-8 -*-



"""
The following code is for training the agents.

To run it:
1) specify the agent you want to train by using one of the following reward functions (set the REWARD constant):

'rewardID_player1' - for ID agent controlling the paddle on the left
'rewardID_player2' - for ID agent controlling the paddle on the right
'rewardSE_player1' - for SE agent controlling the paddle on the left
'rewardSE_player2' - for SE agent controlling the paddle on the right

2) import the version of the pong game corresponding to your chosen reward,
either pong_L or pong_R as pong

Credits:
The following code is a modification of https://github.com/malreddysid/pong_RL
"""



# importing libraries
import pong as pong # change to either pong_L or pong_R to correspond with the reward signal
import tensorflow as tf
import cv2
import numpy as np
import random
import os
from collections import deque


# constants
REWARD = 'rewardID_player1' # change according to which agent you want to train
ACTIONS = 3 # up, down, don't move
GAMMA = 0.99 # learning rate
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORE = 500000  # epsilon annealing
OBSERVE = 50000  # populating memory
USE_MODEL = False
SAVE_STEP = 50000 # generate checkpoints every X timesteps
REPLAY_MEMORY = 500000 # size of memory replay
BATCH = 100 # batch size



# create a tensorflow graph
def createGraph():
     with tf.device('/gpu:0'):
        W_conv1 = tf.Variable(tf.truncated_normal([6, 6, 4, 32], stddev=0.02))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))

        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.02))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))

        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.02))
        b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))

        W_fc4 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.02))
        b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]))

        W_fc5 = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev=0.02))
        b_fc5 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))

        s = tf.placeholder("float", [None, 60, 60, 4])

        conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides = [1, 4, 4, 1], padding = "SAME") + b_conv1)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides = [1, 2, 2, 1], padding = "SAME") + b_conv2)

        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides = [1, 1, 1, 1], padding = "SAME") + b_conv3)

        conv3_flat = tf.reshape(conv3, [-1, 1024])

        fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)

        fc5 = tf.matmul(fc4, W_fc5) + b_fc5

        return s, fc5

# load and train DQN on pixel data
def trainGraph(inp, out):
    # preparation stage - game, frames, saver and checkpoints management
    # to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None]) # ground truth
    global_step = tf.Variable(0, name='global_step')

    # action
    action = tf.reduce_sum(tf.multiply(out, argmax), reduction_indices = 1)
    # cost function which we will reduce through backpropagation
    # what's the action ???
    cost = tf.reduce_mean(tf.square(action - gt))
    # optimization function to minimize our cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # initialize the game
    game = pong.PongGame(0)

    # create a queue for experience replay to store policies
    DL = deque()
    #DR = deque()
    # get intial frame
    frame = game.getPresentFrame()

    # stack frames, create the input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis = 2)

    # saver and checkpoints management
    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 0)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth=True
    sess = tf.InteractiveSession(config=config)
    #sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))

    checkpoint = tf.train.latest_checkpoint('./checkpoints')
    if checkpoint != None:
        print('Restore Checkpoint %s'%(checkpoint))
        saver.restore(sess, checkpoint)
        print("Model restored.")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Initialized new Graph")

    t = global_step.eval()
    c= 0

    epsilon = INITIAL_EPSILON

    train_side = 0
    # training DQN and exporting stats

    while(1):
        # output tensor
        out_t = out.eval(feed_dict = {inp : [inp_t]})[0]
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        # pick action
        if(random.random() <= epsilon and not USE_MODEL):
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE


        if train_side == 1:
            action_tuple = (None, argmax_t)
        else:
            action_tuple = (argmax_t, None)

        # My first goal is to run one agent that can beat the opponent both sides

        frame, rewards, done, cumScores = game.getNextFrame(action_tuple)

        reward_t = rewards[train_side]

        # flip the image data to remian the same view as the left side
        if train_side == 1:
            frame = frame[::-1, :]

        #cv2.imwrite("imgs/%i_train_side.png" % t, frame)

        frame = np.reshape(frame, (60, 60, 1))
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis = 2)
        #if train_side == 1:
        #        DR.append((inp_t, argmax_t, reward_t, inp_t1))
        #        if len(DR) > REPLAY_MEMORY:
        #            DR.popleft()
        #else:
        DL.append((inp_t, argmax_t, reward_t, inp_t1))
        if len(DL) > REPLAY_MEMORY:
            DL.popleft()

        if c > OBSERVE and not USE_MODEL:
            #if train_side == 1:
            #    minibatch = random.sample(DR, BATCH)
            #else:
            minibatch = random.sample(DL, BATCH)
            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict = {inp : inp_t1_batch})

            # add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            # train on that
            train_step.run(feed_dict = {
                           gt : gt_batch,
                           argmax : argmax_batch,
                           inp : inp_batch
                           })

        # update our input tensor the the next frame
        inp_t = inp_t1
        t = t + 1
        c = c + 1

        # save checkints
        if t % SAVE_STEP == 0 and not USE_MODEL:
            sess.run(global_step.assign(t))
            saver.save(sess, './checkpoints/' + 'model.ckpt', global_step=t)

        # save stats log
        if done:
            with open('stats_test.txt', 'a') as log:
                scoreline = 'TIMESTEP ' + str(t) + ' cumScore1 ' + str(cumScores[train_side])+' cumScore2 ' + str(cumScores[1-train_side])+' train side '+str(train_side)+'\n'
                print(scoreline.strip())
                log.write(scoreline)
            train_side = 1 - train_side



def main():
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    inp, out = createGraph()
    trainGraph(inp, out)



if __name__ == "__main__":
    main()
