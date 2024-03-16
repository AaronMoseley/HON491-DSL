import RLModel
import LevelManager
import torch
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = RLModel.WarGamesAI(device).to(device)

levelLoc = "Levels/Testing/"

print("\nAvailable Levels: ")
print(*os.listdir(levelLoc), sep=", ")
levelName = input("Which level do you want to play? ")
while(levelName not in os.listdir(levelLoc)):
    print(*os.listdir(levelLoc), sep=", ")
    levelName = input("Invalid level name. Which level do you want to play? ")

turnCounter = 0
levelState, buildingTurnCounter = LevelManager.readLevel(levelLoc + levelName)

print(model(levelState, -1))