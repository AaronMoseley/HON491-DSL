import math
import random

#0 for ocean
#1 for land
#2 for player 1 start
#3 for player 2 start

#settlement, town, barracks
buildingTurnThresholds = [6, 7, 5]

#chance the attacking solider wins against an enemy solider
attackThreshold = 0.75

def swapPositions(lis, pos1, pos2):
    temp=lis[pos1]
    lis[pos1]=lis[pos2]
    lis[pos2]=temp
    return lis

def readLevel(file):
    f = open(file, "r")

    result = []
    buildingTurnCounter = []
    for line in f:
        currArr = []
        turnCounterArr = []
        for char in line:
            if char == "3":
                currArr.append(-2)
                turnCounterArr.append(-1)
            elif char.isdigit():
                currArr.append(int(char))

                if char == "2":
                    turnCounterArr.append(1)
                else:
                    turnCounterArr.append(0)

        result.append(currArr)
        buildingTurnCounter.append(turnCounterArr)
    
    for i in range(len(result)):
        for j in range(len(result[i])):
            if result[i][j] == 2 or result[i][j] == -2:
                result = spawnUnit(result, i, j)

    return result, buildingTurnCounter

def checkWinCond(levelState):
    hasNeg = False
    hasPos = False
    
    for row in levelState:
        for tile in row:
            if tile < 0 and not hasNeg:
                hasNeg = True
            elif tile > 1 and not hasPos:
                hasPos = True

            if hasNeg and hasPos:
                return 0
            
    if hasPos:
        return 1
    else:
        return -1

def findValidSpawnTile(levelState, i, j):
    offsets = [0, -1, 1]

    for k in offsets:
        if i + k > len(levelState) or i + k < 0:
            continue

        for l in offsets:
            if j + l > len(levelState[i + k]) or j + l < 0:
                continue

            if(levelState[i + k][j + l] == 1):
                return [i + k, j + l]
    
    return None

def spawnUnit(levelState, origin0, origin1):
    validPos = findValidSpawnTile(levelState, origin0, origin1)
    unitID = int(math.copysign(abs(levelState[origin0][origin1]) + 3, levelState[origin0][origin1]))

    if validPos != None:
        levelState[validPos[0]][validPos[1]] = unitID

    return levelState

def incrementTurn(levelState, buildingTurnCounter):
    for i in range(len(buildingTurnCounter)):
        for j in range(len(buildingTurnCounter[i])):
            if buildingTurnCounter[i][j] > 0:
                buildingTurnCounter[i][j] += 1
            elif buildingTurnCounter[i][j] < 0:
                buildingTurnCounter[i][j] -= 1

            if abs(buildingTurnCounter[i][j]) > 0:
                if abs(buildingTurnCounter[i][j]) > buildingTurnThresholds[abs(levelState[i][j]) - 2]:
                    levelState = spawnUnit(levelState, i, j)
                    buildingTurnCounter[i][j] = int(math.copysign(1, buildingTurnCounter[i][j]))

    return levelState, buildingTurnCounter

def isValidMove(levelState, tiles, currPlayer):
    #Out of bounds
    if tiles[0][0] < 0 or tiles[0][0] >= len(levelState[0]) or tiles[1][0] < 0 or tiles[1][0] >= len(levelState[0]):
        return False
    
    if tiles[0][1] < 0 or tiles[0][1] >= len(levelState) or tiles[1][1] < 0 or tiles[1][1] >= len(levelState):
        return False

    #Moving more than 1 space (allows diagonal)
    if abs(tiles[0][0] - tiles[1][0]) > 1 or abs(tiles[0][1] - tiles[1][1]) > 1:
        return False

    #A unit that can be moved
    if abs(levelState[tiles[0][0]][tiles[0][1]]) < 5 or abs(levelState[tiles[0][0]][tiles[0][1]]) > 7:
        return False

    #Can only move a unit of your team
    if levelState[tiles[0][0]][tiles[0][1]] != int(math.copysign(levelState[tiles[0][0]][tiles[0][1]], currPlayer)):
        return False

    #Can't move into your own unit or building, accounts for the ground being positive
    if levelState[tiles[1][0]][tiles[1][1]] == int(math.copysign(levelState[tiles[1][0]][tiles[1][1]], currPlayer)) and levelState[tiles[1][0]][tiles[1][1]] != 1 and (tiles[1][0] - tiles[0][0]) + (tiles[1][1] - tiles[0][1]) != 0:
        return False

    #Can't move into something that's not ground if you're not a soldier
    if levelState[tiles[1][0]][tiles[1][1]] != 1 and abs(levelState[tiles[0][0]][tiles[0][1]]) != 7 and (tiles[1][0] - tiles[0][0]) + (tiles[1][1] - tiles[0][1]) != 0:
        return False

    #Can't move into nothing
    if levelState[tiles[1][0]][tiles[1][1]] == 0:
        return False
    
    if tiles[0][0] != tiles[1][0] and tiles[0][1] != tiles[1][1] and abs(levelState[tiles[0][0]][tiles[0][1]]) == 7:
        return False
    
    return True

#(beginning column-row) (ending column-row)
def parseMove(levelState, moveStr, currPlayer):
    try:
        tiles = [tile.split("-") for tile in moveStr.split()]
    except:
        return None
    
    if len(tiles) != 2:
        return None
    
    if len(tiles[0]) != 2 or len(tiles[1]) != 2:
        return None

    col1 = tiles[0][0].upper()
    col2 = tiles[1][0].upper()

    if len(col1) > 2 or len(col2) > 2 or not col1.isalpha() or not col2.isalpha():
        return None

    if len(col1) == 1:
        tiles[0][0] = ord(col1[0]) - 65
    else:
        tiles[0][0] = ((ord(col1[0]) - 64) * 26) + ord(col1[1]) - 65

    if len(col2) == 1:
        tiles[1][0] = ord(col2[0]) - 65
    else:
        tiles[1][0] = ((ord(col2[0]) - 64) * 26) + ord(col2[1]) - 65

    if not tiles[0][1].isdigit() or not tiles[1][1].isdigit():
        return None
    
    tiles[0][1] = int(tiles[0][1])
    tiles[1][1] = int(tiles[1][1])

    tiles[0] = swapPositions(tiles[0], 0, 1)
    tiles[1] = swapPositions(tiles[1], 0, 1)

    if not isValidMove(levelState, tiles, currPlayer):
        return None

    return tiles

def makeMove(levelState, buildingTurnCounter, move):
    unitID = levelState[move[0][0]][move[0][1]]
    moveToID = levelState[move[1][0]][move[1][1]]

    if abs(unitID) == 7 and abs(moveToID) == 7:
        chance = random.uniform()
        if chance <= attackThreshold:
            levelState[move[0][0]][move[0][1]] = 1
            levelState[move[1][0]][move[1][1]] = unitID
        else:
            levelState[move[0][0]][move[0][1]] = 1
    elif move[0][0] == move[1][0] and move[0][1] == move[1][1]:
        levelState[move[0][0]][move[0][1]] = int(math.copysign(abs(unitID) - 2, unitID))
        buildingTurnCounter[move[0][0]][move[0][1]] = int(math.copysign(1, unitID))
    else:
        levelState[move[0][0]][move[0][1]] = 1
        levelState[move[1][0]][move[1][1]] = unitID
        buildingTurnCounter[move[0][0]][move[0][1]] = 0
        buildingTurnCounter[move[1][0]][move[1][1]] = 0
    
    return levelState, buildingTurnCounter