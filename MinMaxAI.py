import LevelManager
import math

#settlement, town, barracks, settler, worker, soldier
friendlyUnitValues = [4, 5, 6, 2, 3, 4]
enemyUnitValues = [-4, -5, -6, -2, -3, -4]

def getValidMoves(levelState, currPlayer):
    result = []
    
    for i, row in enumerate(levelState):
        for j, tile in enumerate(row):
            if tile == 1 or tile == 0 or currPlayer != math.copysign(currPlayer, tile):
                continue

            if abs(tile) >= 2 and abs(tile) <= 4:
                continue

            print(tile)

            offsets = [-1, 0, 1]
            for offset0 in offsets:
                for offset1 in offsets:
                    if offset0 + offset1 == 0 and abs(tile) == 7:
                        continue

                    currMove = [[i, j], [i + offset0, j + offset1]]
                    if LevelManager.isValidMove(levelState, currMove, currPlayer):
                        result.append(currMove)
                    else:
                        print("invalid")

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