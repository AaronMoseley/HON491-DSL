import LevelManager
import math
import copy
import random
import time

#settlement, town, barracks, settler, worker, soldier
friendlyUnitValues = [0, 3, 1, 0, 0.5, 0.8]
enemyUnitValues = [0, -2.5, -2, 0, -1, -1.5]

def chooseMove(levelState, buildingTurnCounter, currPlayer, stateBudget=1000):
    validMoves = getValidMoves(levelState, currPlayer)
    
    if len(validMoves) == 0:
        return None
    
    random.seed(time.time())
    random.shuffle(validMoves)

    for i in range(len(validMoves)):
        if isCreateOrCapture(levelState, validMoves[i]):
            validMoves.insert(0, validMoves.pop(i))

    currDepth = 2

    finalMove = None

    while stateBudget > 0:
        bestUtil = -float("inf")
        bestMove = None
        alpha = -float("inf")
        beta = float("inf")
        for move in validMoves:
            newState = copy.deepcopy(levelState)
            newTurnCounter = copy.deepcopy(buildingTurnCounter)

            newState, newTurnCounter = LevelManager.makeMove(newState, newTurnCounter, move)
            newState, newTurnCounter = LevelManager.incrementTurn(newState, newTurnCounter)

            util, stateBudget = minimax(newState, newTurnCounter, 2, currDepth, True, alpha, beta, currPlayer * -1, currPlayer, stateBudget)

            if util == None or stateBudget <= -1:
                return finalMove

            if util > bestUtil:
                bestUtil = util
                bestMove = move
                alpha = max(alpha, bestUtil)

            currDepth += 1
            finalMove = bestMove

    if finalMove == None:
        #return validMoves[0]
        return random.choice(validMoves)

    return finalMove

def minimax(currState, currTurnCounter, currDepth, maxDepth, isMaximizing, alpha, beta, currPlayer, startingPlayer, stateBudget):
    if stateBudget <= 0:
        return None, -1
    
    if LevelManager.checkWinCond(currState) != 0:
        return (float("inf"), stateBudget - 1) if startingPlayer == math.copysign(startingPlayer, LevelManager.checkWinCond(currState)) else (float("-inf"), stateBudget - 1)
    
    if currDepth >= maxDepth:
        return getValueOfState(currState, startingPlayer), stateBudget - 1

    validMoves = getValidMoves(currState, currPlayer)
    if len(validMoves) == 0:
        newState = copy.deepcopy(currState)
        newTurnCounter = copy.deepcopy(currTurnCounter)

        newState, newTurnCounter = LevelManager.incrementTurn(newState, newTurnCounter)
        return minimax(newState, newTurnCounter, currDepth, maxDepth, not isMaximizing, alpha, beta, currPlayer * -1, startingPlayer, stateBudget - 1)
    
    #random.shuffle(validMoves)

    for i in range(len(validMoves)):
        if isCreateOrCapture(currState, validMoves[i]):
            validMoves.insert(0, validMoves.pop(i))

    if isMaximizing:
        bestVal = -float("inf")
        for move in validMoves:
            stateBudget -= 1
            if stateBudget <= 0:
                return None, -1
            
            childState = copy.deepcopy(currState)
            childTurnCounter = copy.deepcopy(currTurnCounter)

            childState, childTurnCounter = LevelManager.makeMove(childState, childTurnCounter, move)
            childState, childTurnCounter = LevelManager.incrementTurn(childState, childTurnCounter)

            util, stateBudget= minimax(childState, childTurnCounter, currDepth + 1, maxDepth, False, alpha, beta, currPlayer * -1, startingPlayer, stateBudget)

            if util == None or stateBudget <= 0:
                return None, -1

            bestVal = max(bestVal, util)

            if bestVal > beta:
                break

            alpha = max(alpha, bestVal)

        return bestVal, stateBudget
    else:
        bestVal = float("inf")
        for move in validMoves:
            stateBudget -= 1
            if stateBudget <= 0:
                return None, -1
            
            childState = copy.deepcopy(currState)
            childTurnCounter = copy.deepcopy(currTurnCounter)

            childState, childTurnCounter = LevelManager.makeMove(childState, childTurnCounter, move)
            childState, childTurnCounter = LevelManager.incrementTurn(childState, childTurnCounter)

            util, stateBudget = minimax(childState, childTurnCounter, currDepth + 1, maxDepth, True, alpha, beta, currPlayer * -1, startingPlayer, stateBudget)

            if util == None or stateBudget <= 0:
                return None, -1

            bestVal = min(bestVal, util)

            if bestVal < alpha:
                break

            beta = min(beta, bestVal)

        return bestVal, stateBudget

def isCreateOrCapture(levelState, move):
    if move[0][0] == move[1][0] and move[1][0] == move[1][1]:
        return True
    
    if levelState[move[1][0]][move[1][1]] != 1:
        return True
    
    return False

def getValidMoves(levelState, currPlayer):
    result = []
    
    for i, row in enumerate(levelState):
        for j, tile in enumerate(row):
            if tile == 1 or tile == 0 or currPlayer != math.copysign(currPlayer, tile):
                continue

            if abs(tile) >= 2 and abs(tile) <= 4:
                continue

            offsets = [-1, 0, 1]
            for offset0 in offsets:
                for offset1 in offsets:
                    if offset0 == 0 and offset1 == 0 and abs(tile) == 7:
                        continue

                    currMove = [[i, j], [i + offset0, j + offset1]]
                    if LevelManager.isValidMove(levelState, currMove, currPlayer):
                        result.append(currMove)

    return result

def getValueOfState(levelState, currPlayer):
    if LevelManager.checkWinCond(levelState) != 0:
        return float("inf") if currPlayer == math.copysign(currPlayer, LevelManager.checkWinCond(levelState)) else -float("inf")
    
    result = 0
    
    for row in levelState:
        for tile in row:
            if tile == 0 or tile == 1:
                continue

            if tile == math.copysign(tile, currPlayer):
                result += friendlyUnitValues[abs(tile) - 2]
            else:
                result += enemyUnitValues[abs(tile) - 2]

    return result