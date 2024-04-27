import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
import os

noise1 = PerlinNoise(octaves=3)
noise2 = PerlinNoise(octaves=6)
noise3 = PerlinNoise(octaves=12)
noise4 = PerlinNoise(octaves=24)

xpix, ypix = 8, 8
pic = []
for i in range(xpix):
    row = []
    for j in range(ypix):
        noise_val = noise1([i/xpix, j/ypix])
        noise_val += 0.5 * noise2([i/xpix, j/ypix])
        noise_val += 0.25 * noise3([i/xpix, j/ypix])
        noise_val += 0.125 * noise4([i/xpix, j/ypix])

        row.append(noise_val)
    pic.append(row)

threshold = -0.2
newPic = []
for i in range(xpix):
    row = []
    for j in range(ypix):
        row.append(1 if pic[i][j] > threshold else 0)
    newPic.append(row)

fileLoc = os.path.dirname(__file__) + "\\Levels\\Final\\Unseen\\"
fileName = "Level7.txt"

f = open(fileLoc + fileName, "w")
for i in range(xpix):
    for j in range(ypix):
        f.write(str(newPic[i][j]))
    f.write("\n")

f.close()

plt.imshow(newPic, cmap='gray')
plt.show()