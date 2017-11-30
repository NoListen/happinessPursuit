# -*- coding: utf-8 -*-
import math


"""
The following code specifies the reward signals provided to the agents
in the following order:
1) reward corresponding to ID values
2) reward corresponding to SE values defined for agent controlling the
paddle on the left or right
"""


# zero sum game.
def rewardID(score1, score2):
    """ computes ID reward, +1 if player scores a point, -1 otherwise """
    reward = score1 - score2
    return reward, -reward


def unitRewardSE(score_diff, diff):
    # winner
    if score_diff == 0:
        reward = 0
    elif score_diff > 0:
        if diff <= 1:
            reward = 1
        else:
            reward = 1.0/diff
    else:
        if diff == 0:
            reward = -0.05
        else:
            reward = abs(diff/11.0)

    return reward

def rewardSE(score1, score2, cum_score1, cum_score2):
    """ computes SE reward, depending on whether the agent is 'catching up'
    or 'getting ahead' """
    score_diff = score1 - score2
    diff = cum_score1 - cum_score2
    return unitRewardSE(score_diff, diff), unitRewardSE(-score_diff, diff)
