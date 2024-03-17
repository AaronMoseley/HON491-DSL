import LevelManager
import Display
import math
import copy

#settlement, town, barracks, settler, worker, soldier
friendlyUnitValues = [20, 10, 15, 5, 11, 20]
enemyUnitValues = [-30, -15, -20, -5, -10, -20]

def chooseMove(levelState, buildingTurnCounter, currPlayer, stateBudget=1000):
    if len(getValidMoves(levelState, currPlayer)) == 0:
        return None
    
    bestMove = None
    maxUtil = -float("inf")
    alpha = -float("inf")
    beta = float("inf")

    currDepth = 2

    while stateBudget > 0:
        currBest = None
        currMaxUtil = -float("inf")

        for move in getValidMoves(levelState, currPlayer):
            stateBudget -= 1
            if stateBudget <= 0:
                break
            
            childState = copy.deepcopy(levelState)
            childTurnCounter = copy.deepcopy(buildingTurnCounter)

            childState, childTurnCounter = LevelManager.makeMove(childState, childTurnCounter, move)
            childState, childTurnCounter = LevelManager.incrementTurn(childState, childTurnCounter)

            util = 0

            util, stateBudget, alpha, beta = findMin(childState, childTurnCounter, alpha, beta, stateBudget, currDepth, 1, -1 * currPlayer, currPlayer)
            
            if(util > maxUtil):
                currMaxUtil = util
                currBest = move

        if currMaxUtil >= maxUtil:
            maxUtil = currMaxUtil
            bestMove = currBest

        currDepth += 1

        if stateBudget <= 0:
            break

    print(currDepth)
    return bestMove

def findMin(curr, buildingTurnCounter, alpha, beta, stateBudget, maxDepth, currDepth, player, startPlayer):
    if currDepth >= maxDepth:
        return getValueOfState(curr, startPlayer), stateBudget, alpha, beta

    curr = copy.deepcopy(curr)
    buildingTurnCounter = copy.deepcopy(buildingTurnCounter)

    validMoves = getValidMoves(curr, player)
    if len(validMoves) == 0:
        curr, buildingTurnCounter = LevelManager.incrementTurn(curr, buildingTurnCounter)
        return findMax(curr, buildingTurnCounter, alpha, beta, stateBudget - 1, maxDepth, currDepth + 1, player * -1, startPlayer)

    minUtil = float("inf")

    for move in validMoves:
        stateBudget -= 1
        if stateBudget <= 0:
            break

        childState, childTurnCounter = LevelManager.makeMove(curr, buildingTurnCounter, move)
        childState, childTurnCounter = LevelManager.incrementTurn(childState, childTurnCounter)

        childVal, stateBudget, alpha, beta = findMax(childState, childTurnCounter, alpha, beta, stateBudget, maxDepth, currDepth + 1, player * -1, startPlayer)
        
        minUtil = min(childVal, minUtil)

        if minUtil <= alpha:
            break
        
        beta = min(beta, minUtil)

    return minUtil, stateBudget, alpha, beta

def findMax(curr, buildingTurnCounter, alpha, beta, stateBudget, maxDepth, currDepth, player, startPlayer):
    if currDepth >= maxDepth:
        return getValueOfState(curr, startPlayer), stateBudget, alpha, beta

    curr = copy.deepcopy(curr)
    buildingTurnCounter = copy.deepcopy(buildingTurnCounter)

    maxUtil = -float("inf")

    validMoves = getValidMoves(curr, player)

    if len(validMoves) == 0:
        curr, buildingTurnCounter = LevelManager.incrementTurn(curr, buildingTurnCounter)
        return findMin(curr, buildingTurnCounter, alpha, beta, stateBudget - 1, maxDepth, currDepth + 1, player * -1, startPlayer)

    for move in validMoves:
        stateBudget -= 1
        if stateBudget <= 0:
            break

        childState, childTurnCounter = LevelManager.makeMove(curr, buildingTurnCounter, move)
        childState, childTurnCounter = LevelManager.incrementTurn(childState, childTurnCounter)

        childVal, stateBudget, alpha, beta = findMin(childState, childTurnCounter, alpha, beta, stateBudget, maxDepth, currDepth + 1, player * -1, startPlayer)

        maxUtil = max(childVal, maxUtil)

        if maxUtil >= beta:
            break
        
        alpha = max(maxUtil, alpha)
    
    return maxUtil, stateBudget, alpha, beta

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