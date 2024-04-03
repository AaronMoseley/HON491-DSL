import os

def validSelection(userIn):
    if len(userIn.split()) != 2:
        return False
    
    for player in userIn.split():
        if not player.isnumeric():
            return False
        
        if int(player) < 0 or int(player) > 3:
            return False
    
    return True

def printMainMenu():
    os.system("cls")
    print("Welcome to War Games. Please select the player types as \"[player1Type] [player2Type]\" from the list below. Type \"quit\" to quit and \"help\" for the help menu.")
    print("0. Human Player")
    print("1. Minimax Bot")
    print("2. Reinforcement Learning Bot")
    print("3. Random Bot")

def printHelpMenu():
    os.system("cls")
    print("This is the help menu")
    print("Type anything to return to the menu")