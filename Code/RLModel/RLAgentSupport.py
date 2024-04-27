import LevelManager
import math
import copy
import random
import time
import torch

#friendlyUnitValues = [0.4, 0.2, 0.35, 0.1, 0.15, 0.3]
#enemyUnitValues = [-0.55, -0.25, -0.4, -0.15, -0.15, -0.35]

#friendlyUnitValues = [0, 0.6, 0.3, 0, 0.2, 0.25]
#enemyUnitValues = [0, -0.5, -0.4, 0, -0.1, -0.3]

friendlyUnitValues = [0, 9, 6, 0, 4, 3]
enemyUnitValues = [0, -9, -6, 0, -4, -3]

def MSELoss(actual, target, device):
    loss = torch.zeros(1).to(device)
    loss.requires_grad = True
    for i, el in enumerate(actual):
        loss = loss + torch.pow(el - target[i], 2)

    loss = loss / target.size(dim=0)

    return loss

"""
def reward(currState, prevState, currPlayer):
    gameEndVal = 100
    
    if LevelManager.checkWinCond(currState) != 0:
        return torch.tensor(gameEndVal, dtype=torch.int8) if currPlayer == math.copysign(currPlayer, LevelManager.checkWinCond(currState)) else torch.tensor(-gameEndVal, dtype=torch.int8)
    
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
    return currUtil - prevUtil
"""

def reward(levelState, currPlayer):
    gameEndVal = 500

    if LevelManager.checkWinCond(levelState) != 0:
        return torch.tensor(gameEndVal, dtype=torch.int8) if currPlayer == math.copysign(currPlayer, LevelManager.checkWinCond(levelState)) else torch.tensor(-gameEndVal, dtype=torch.int8)
    
    result = torch.zeros(1)
    numUnits = 1

    for row in levelState:
        for tile in row:
            if tile == 0 or tile == 1:
                continue

            if tile == math.copysign(tile, currPlayer):
                result += friendlyUnitValues[abs(tile) - 2]
            else:
                result += enemyUnitValues[abs(tile) - 2]
                numUnits += 1

    return result / numUnits

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
            newTurnCounter[i][j] = sign

    return newState, newTurnCounter