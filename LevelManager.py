import math
import random
import copy

#0 for ocean
#1 for land
#2 for player 1 start
#3 for player 2 start

#settlement, town, barracks
buildingTurnThresholds = [14, 12, 10]

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
    newState = copy.deepcopy(levelState)

    validPos = findValidSpawnTile(newState, origin0, origin1)
    unitID = int(math.copysign(abs(newState[origin0][origin1]) + 3, newState[origin0][origin1]))

    if validPos != None:
        newState[validPos[0]][validPos[1]] = unitID

    return newState

def incrementTurn(levelState, buildingTurnCounter):
    newState = copy.deepcopy(levelState)
    newTurnCounter = copy.deepcopy(buildingTurnCounter)

    for i in range(len(newTurnCounter)):
        for j in range(len(newTurnCounter[i])):
            if newTurnCounter[i][j] > 0:
                newTurnCounter[i][j] += 1
            elif newTurnCounter[i][j] < 0:
                newTurnCounter[i][j] -= 1

            if abs(newTurnCounter[i][j]) > 0:
                if abs(newTurnCounter[i][j]) > buildingTurnThresholds[abs(newState[i][j]) - 2]:
                    newState = spawnUnit(newState, i, j)
                    newTurnCounter[i][j] = int(math.copysign(1, newTurnCounter[i][j]))

    return newState, newTurnCounter

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
    if levelState[tiles[1][0]][tiles[1][1]] == int(math.copysign(levelState[tiles[1][0]][tiles[1][1]], currPlayer)) and levelState[tiles[1][0]][tiles[1][1]] != 1 and (tiles[1][0] - tiles[0][0] != 0 or tiles[1][1] - tiles[0][1] != 0):
        return False

    #Can't move into something that's not ground if you're not a soldier
    if levelState[tiles[1][0]][tiles[1][1]] != 1 and abs(levelState[tiles[0][0]][tiles[0][1]]) != 7 and (tiles[1][0] - tiles[0][0] != 0 or tiles[1][1] - tiles[0][1] != 0):
        return False

    #Can't move into nothing
    if levelState[tiles[1][0]][tiles[1][1]] == 0:
        return False
    
    if tiles[0][0] == tiles[1][0] and tiles[0][1] == tiles[1][1] and abs(levelState[tiles[0][0]][tiles[0][1]]) == 7:
        return False
    
    return True

#(beginning column-row) (ending column-row)
def parseMove(levelState, moveStr, currPlayer):
    try:
        moveStr = moveStr.split()
    except:
        return None
    
    if len(moveStr) != 2:
        return None
    
    tiles = [[], []]
    for i, tile in enumerate(moveStr):
        firstNum = 0
        for j, c in enumerate(tile):
            if c.isdigit():
                firstNum = j
                break

        tiles[i].append(tile[:firstNum])
        tiles[i].append(tile[firstNum:])

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
    newState = copy.deepcopy(levelState)
    newTurnCounter = copy.deepcopy(buildingTurnCounter)

    if move == None:
        return newState, newTurnCounter
    
    unitID = levelState[move[0][0]][move[0][1]]
    moveToID = levelState[move[1][0]][move[1][1]]

    if abs(unitID) == 7 and abs(moveToID) == 7:
        chance = random.uniform(0, 1)
        if chance <= attackThreshold:
            newState[move[0][0]][move[0][1]] = 1
            newState[move[1][0]][move[1][1]] = unitID
        else:
            newState[move[0][0]][move[0][1]] = 1
    elif move[0][0] == move[1][0] and move[0][1] == move[1][1]:
        newState[move[0][0]][move[0][1]] = int(math.copysign(abs(unitID) - 2, unitID))
        newTurnCounter[move[0][0]][move[0][1]] = int(math.copysign(1, unitID))
    else:
        newState[move[0][0]][move[0][1]] = 1
        newState[move[1][0]][move[1][1]] = unitID
        newTurnCounter[move[0][0]][move[0][1]] = 0
        newTurnCounter[move[1][0]][move[1][1]] = 0
    
    return newState, newTurnCounter