import LevelManager
import math
import copy
import random
import time
import torch

friendlyUnitValues = [0.4, 0.2, 0.35, 0.1, 0.15, 0.3]
enemyUnitValues = [-0.55, -0.25, -0.4, -0.15, -0.15, -0.35]

def rlLoss(levelState, currPlayer):
    lostGameVal = -5

    if LevelManager.checkWinCond(levelState) != 0:
        return 0 if currPlayer == math.copysign(currPlayer, LevelManager.checkWinCond(levelState)) else lostGameVal
    
    result = torch.zeros(1)
    
    for row in levelState:
        for tile in row:
            if tile == 0 or tile == 1:
                continue

            if tile == math.copysign(tile, currPlayer):
                result += friendlyUnitValues[abs(tile) - 2]
            else:
                result += enemyUnitValues[abs(tile) - 2]

    result.requires_grad = True

    sigm = torch.nn.Sigmoid()
    return sigm(-result)

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
            newState[i][j] = sign * random.randint(3, 7)

            if abs(newState[i][j]) >= 2 and abs(newState[i][j]) <= 4:
                newTurnCounter[i][j] = sign

    return newState, newTurnCounter