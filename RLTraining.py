import RLAgentSupport
import RLModel
import MinMaxAI
import LevelManager
import torch
import Display
import copy
import random
import time
import NewRLModel
import os

random.seed(time.time())

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#model = RLModel.WarGamesAI(device).to(device)

#targetNet = RLModel.WarGamesAI(device).to(device)

model = NewRLModel.MinMaxWarGamesAI(device).to(device)
targetNet = NewRLModel.MinMaxWarGamesAI(device).to(device)

targetNet.load_state_dict(model.state_dict())

levelLoc = "Levels/Final/"
#levelName = "Level1.txt"

numEpochs = 1000000
learnRate = 0.05
model.train()

maxNumTurns = 50
selfPlayThreshold = 25

mutationPower = 0.01
targetStartUnits = 10

modelSavePath = "Models/"
modelName = "TestRLModel10"
checkpointEpoch = 10

currIterations = 0
iterationsToResetTarget = 100

minReplayLen = 15
maxReplayLen = 1500
batchSize = 5
replay = []
replayEntry = []

cosineAnnealingRestartEpochs = 10

optimizer = torch.optim.Adam(model.parameters(), lr=learnRate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, cosineAnnealingRestartEpochs, T_mult=1)

totalLoss = 0

gamesWon = 0
gamesLost = 0
print("Beginning training")
for epoch in range(numEpochs):
    levelName = random.choice(os.listdir(levelLoc))

    levelState, buildingTurnCounter = LevelManager.readLevel(levelLoc + levelName, spawnUnits=False)

    if random.uniform(0, 1) <= 0.80:
        levelState, buildingTurnCounter = RLAgentSupport.randomizeState(levelState, buildingTurnCounter, targetStartUnits / (len(levelState) * len(levelState[0])))

    if random.uniform(0, 1) > 0.5:
        levelState.reverse()
        buildingTurnCounter.reverse()

        for i in range(len(levelState)):
            for j in range(len(levelState[i])):
                if abs(levelState[i][j]) > 1:
                    levelState[i][j] *= -1
                    buildingTurnCounter[i][j] *= -1

    modelPlayer = 1
    opposingPlayerType = 1 if epoch % 2 == 0 else 2
    #opposingPlayerType = 2

    if opposingPlayerType == 2:
        #mutatedNetwork = RLModel.WarGamesAI(device).to(device)
        mutatedNetwork = NewRLModel.MinMaxWarGamesAI(device).to(device)
        for param in mutatedNetwork.parameters():
            param.data += mutationPower * torch.randn_like(param)

    turnCounter = 0
    currPlayer = -1
    while LevelManager.checkWinCond(levelState) == 0 and turnCounter <= maxNumTurns:
        currPlayer *= -1
        levelState, buildingTurnCounter = LevelManager.incrementTurn(levelState, buildingTurnCounter)

        if currPlayer == -1:
            turnCounter += 1

        parsedMove = None
        if len(MinMaxAI.getValidMoves(levelState, currPlayer)) > 0:
            if currPlayer == modelPlayer:
                #parsedMove = model([levelState], modelPlayer, outputMove=True)
                parsedMove = model([levelState], [buildingTurnCounter], modelPlayer, returnMove=True)
                if parsedMove != None:
                    parsedMove = parsedMove[0]
                #print(parsedMove)
            else:
                if opposingPlayerType == 1:
                    parsedMove = MinMaxAI.chooseMove(levelState, buildingTurnCounter, currPlayer, stateBudget=10000)
                else:
                    reverseState = copy.deepcopy(levelState)
                    reverseState.reverse()

                    reverseTurnCounter = copy.deepcopy(buildingTurnCounter)
                    reverseTurnCounter.reverse()

                    for i in range(len(reverseState)):
                        for j in range(len(reverseState[i])):
                            if abs(reverseState[i][j]) > 1:
                                reverseState[i][j] *= -1
                                reverseTurnCounter[i][j] *= -1

                    #parsedMove = mutatedNetwork([levelState], modelPlayer, outputMove=True)
                    parsedMove = mutatedNetwork([reverseState], [reverseTurnCounter], modelPlayer, returnMove=True)
                    if parsedMove != None:
                        parsedMove = parsedMove[0]
                        if parsedMove != None:
                            parsedMove[0][0] = 7 - parsedMove[0][0]
                            parsedMove[1][0] = 7 - parsedMove[1][0]

        if parsedMove != None:
            newState = copy.deepcopy(levelState)
            newTurnCounter = copy.deepcopy(buildingTurnCounter)

            newState, newTurnCounter = LevelManager.makeMove(newState, newTurnCounter, parsedMove)
            
            if currPlayer == modelPlayer:
                optimizer.zero_grad()
                #reward = RLAgentSupport.reward(newState, levelState, modelPlayer)

                #print(reward)

                #replayEntry = [levelState, parsedMove, reward]
                replayEntry = [levelState, buildingTurnCounter, parsedMove]

                if len(replay) >= minReplayLen:
                    batch = []
                    for _ in range(batchSize):
                        batch.append(random.choice(replay))

                    batchStates = [x[0] for x in batch]
                    batchCounters = [x[1] for x in batch]

                    qVals = model(batchStates, batchCounters, modelPlayer, returnMove=False)
                    #qVals = model(batchStates, modelPlayer, outputMove=False)

                    batchNextStates = [x[4] for x in batch]
                    batchNextCounters = [x[5] for x in batch]

                    #targetModelResult = targetNet(batchNextStates, modelPlayer, outputMove=False)
                    targetModelResult = targetNet(batchNextStates, batchNextCounters, modelPlayer, returnMove=False)
                    targetQvals = torch.Tensor([targetModelResult[i] + batch[i][3].to(device) for i in range(batchSize)])

                    loss = RLAgentSupport.MSELoss(qVals, targetQvals, device)

                    totalLoss += loss.cpu().detach().numpy()

                    loss.backward()
                    optimizer.step()
                    currIterations += 1

                    #if currIterations >= iterationsToResetTarget:
                    #    targetNet.load_state_dict(model.state_dict())
                    #    currIterations = 0

                print("Game: " + str(epoch))
                Display.displayLevel(levelState, turnCounter, False)
            else:
                if len(replayEntry) > 0:
                    reward = RLAgentSupport.reward(newState, replayEntry[0], modelPlayer)
                    replayEntry.append(reward)

                    replayEntry.append(newState)
                    replayEntry.append(newTurnCounter)
                    replay.append(copy.deepcopy(replayEntry))
                    replayEntry = []

                    if len(replay) > maxReplayLen:
                        replay.pop(random.randrange(len(replay)))
            
            levelState = copy.deepcopy(newState)
            buildingTurnCounter = copy.deepcopy(newTurnCounter)

    scheduler.step(epoch)
    targetNet.load_state_dict(model.state_dict())
    if currIterations != 0:
        print("Epoch " + str(epoch) + " Average Loss: " + str(totalLoss / currIterations))

    currIterations = 0
    totalLoss = 0

    print("Turns taken: " + str(turnCounter))

    winCond = LevelManager.checkWinCond(levelState)
    if winCond == modelPlayer:
        gamesWon += 1
    elif winCond == modelPlayer * -1:
        gamesLost += 1

    print("Record: " + str(gamesWon) + " - " + str(gamesLost) + " / " + str(epoch + 1))

    if epoch % checkpointEpoch == 0:
        torch.save(model.state_dict(), modelSavePath + modelName + "Epoch" + str(epoch))

torch.save(model.state_dict(), modelSavePath + modelName)