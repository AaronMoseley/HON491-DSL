import LevelManager
import math
import copy
import random
import time
import torch

#friendlyUnitValues = [0.4, 0.2, 0.35, 0.1, 0.15, 0.3]
#enemyUnitValues = [-0.55, -0.25, -0.4, -0.15, -0.15, -0.35]

friendlyUnitValues = [0, 0.4, 0.3, 0, 0.2, 0.3]
enemyUnitValues = [0, -0.3, -0.4, 0, -0.2, -0.3]

def MSELoss(actual, target, device):
    loss = torch.zeros(1).to(device)
    loss.requires_grad = True
    for i, el in enumerate(actual):
        loss = loss + torch.pow(el - target[i], 2)

    loss = loss / target.size(dim=0)

    return loss

"""
def rlLoss(currState, prevState, currPlayer):
    lostGameVal = 1

    if LevelManager.checkWinCond(currState) != 0:
        return 0 if currPlayer == math.copysign(currPlayer, LevelManager.checkWinCond(currState)) else lostGameVal
    
    prevUtil = torch.zeros(1)
    for row in prevState:
        for tile in row:
            if tile == 0 or tile == 1:
                continue

            if tile == math.copysign(tile, currPlayer):
                prevUtil += friendlyUnitValues[abs(tile) - 2]
            else:
                prevUtil += enemyUnitValues[abs(tile) - 2]

    currUtil = torch.zeros(1)
    for row in currState:
        for tile in row:
            if tile == 0 or tile == 1:
                continue

            if tile == math.copysign(tile, currPlayer):
                currUtil += friendlyUnitValues[abs(tile) - 2]
            else:
                currUtil += enemyUnitValues[abs(tile) - 2]

    sigm = torch.nn.Sigmoid()
    result = sigm(-(currUtil - prevUtil + 1.5))
    result.requires_grad = True
    return result
"""

def reward(levelState, currPlayer):
    if LevelManager.checkWinCond(levelState) != 0:
        return float("inf") if currPlayer == math.copysign(currPlayer, LevelManager.checkWinCond(levelState)) else -float("inf")
    
    result = torch.zeros(1)
    
    for row in levelState:
        for tile in row:
            if tile == 0 or tile == 1:
                continue

            if tile == math.copysign(tile, currPlayer):
                result += friendlyUnitValues[abs(tile) - 2]
            else:
                result += enemyUnitValues[abs(tile) - 2]

    sigm = torch.nn.Sigmoid()

    return sigm(result)

def randomizeState(levelState, buildingTurnCounter, addUnitProb):
    newState = copy.deepcopy(levelState)
    newTurnCounter = copy.deepcopy(buildingTurnCounter)

    random.seed(time.time())

    for i in range(len(newState)):
        for j in range(len(newState[i])):
            if newState[i][j] != 1:
                continue

            if random.random() > addUnitProb:
                continue

            rowProgress = i / (len(newState) - 1)
            sign = 1 if random.random() < rowProgress else -1
            unit = random.randint(4, 6)
            if unit > 4:
                unit += 1
            newState[i][j] = sign * unit

            if abs(newState[i][j]) >= 2 and abs(newState[i][j]) <= 4:
                newTurnCounter[i][j] = sign

    return newState, newTurnCounter