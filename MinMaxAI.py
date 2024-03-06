import LevelManager
import Display
import math
import copy

#settlement, town, barracks, settler, worker, soldier
friendlyUnitValues = [8, 3, 7, 2, 4, 6]
enemyUnitValues = [-10, -3, -8, -2, -4, -8]

def chooseMove(levelState, buildingTurnCounter, currPlayer):
    stateBudget = 1000
    
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

            util, stateBudget, alpha, beta = findMin(childState, childTurnCounter, alpha, beta, stateBudget, currDepth, 1, -1 * currPlayer)
            
            if(util > maxUtil):
                currMaxUtil = util
                currBest = move

        if currMaxUtil >= maxUtil:
            maxUtil = currMaxUtil
            bestMove = currBest

        currDepth += 1

        if stateBudget <= 0:
            break

    if bestMove != None:
        return bestMove
    else:
        print("error finding best move")
        return getValidMoves(levelState, currPlayer)[0]

def findMin(curr, buildingTurnCounter, alpha, beta, stateBudget, maxDepth, currDepth, player):
    if currDepth >= maxDepth:
        return getValueOfState(curr, player), stateBudget, alpha, beta

    curr = copy.deepcopy(curr)
    buildingTurnCounter = copy.deepcopy(buildingTurnCounter)

    minUtil = float("inf")

    for move in getValidMoves(curr, player):
        stateBudget -= 1
        if stateBudget <= 0:
            break

        childState, childTurnCounter = LevelManager.makeMove(curr, buildingTurnCounter, move)
        childState, childTurnCounter = LevelManager.incrementTurn(childState, childTurnCounter)

        childVal, stateBudget, alpha, beta = findMax(childState, childTurnCounter, alpha, beta, stateBudget, maxDepth, currDepth + 1, player * -1)
        
        minUtil = min(childVal, minUtil)

        if minUtil <= alpha:
            break
        
        beta = min(beta, minUtil)

    return minUtil, stateBudget, alpha, beta

def findMax(curr, buildingTurnCounter, alpha, beta, stateBudget, maxDepth, currDepth, player):
    if currDepth >= maxDepth:
        return getValueOfState(curr, player), stateBudget, alpha, beta

    curr = copy.deepcopy(curr)
    buildingTurnCounter = copy.deepcopy(buildingTurnCounter)

    maxUtil = float("inf")

    for move in getValidMoves(curr, player):
        stateBudget -= 1
        if stateBudget <= 0:
            break

        childState, childTurnCounter = LevelManager.makeMove(curr, buildingTurnCounter, move)
        childState, childTurnCounter = LevelManager.incrementTurn(childState, childTurnCounter)

        childVal, stateBudget, alpha, beta = findMin(childState, childTurnCounter, alpha, beta, stateBudget, maxDepth, currDepth + 1, player * -1)

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
                    if offset0 + offset1 == 0 and abs(tile) == 7:
                        continue

                    currMove = [[i, j], [i + offset0, j + offset1]]
                    if LevelManager.isValidMove(levelState, currMove, currPlayer):
                        result.append(currMove)

    return result

def getValueOfState(levelState, currPlayer):
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