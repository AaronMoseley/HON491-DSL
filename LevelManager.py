#0 for ocean
#1 for land
#2 for player 1 start
#3 for player 2 start

def readLevel(file):
    f = open(file, "r")

    result = []
    for line in f:
        currArr = []
        for char in line:
            if char == "3":
                currArr.append(-2)
            elif char.isdigit():
                currArr.append(int(char))

        result.append(currArr)
    
    return result