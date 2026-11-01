# Install once if needed:
# pip install pandas numpy matplotlib scikit-learn
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("datasets")
if not DATA.exists():
    DATA = Path("../datasets")

import random

maze = pd.read_csv(DATA / "maze.csv", header=None).values
rows, cols = maze.shape
start, goal = (0,0), (rows-1, cols-1)
actions = [(-1,0),(1,0),(0,-1),(0,1)]
Q = np.zeros((rows*cols, 4))
alpha, gamma, epsilon = 0.15, 0.90, 0.15

def move(state, action):
    r,c = state; dr,dc = actions[action]
    nr,nc = r+dr,c+dc
    if nr<0 or nr>=rows or nc<0 or nc>=cols or maze[nr,nc]==1:
        return state, -5
    if (nr,nc)==goal:
        return (nr,nc), 20
    return (nr,nc), -1

for episode in range(1500):
    state = start
    for step in range(100):
        s = state[0]*cols+state[1]
        action = random.randrange(4) if random.random()<epsilon else int(np.argmax(Q[s]))
        next_state, reward = move(state, action)
        ns = next_state[0]*cols+next_state[1]
        Q[s,action] += alpha*(reward + gamma*np.max(Q[ns]) - Q[s,action])
        state = next_state
        if state == goal: break

state=start; path=[state]
for _ in range(rows*cols*3):
    s=state[0]*cols+state[1]
    # choose best currently valid action to avoid getting stuck at a wall
    choices=[]
    for a in range(4):
        nxt, rew=move(state,a)
        if nxt != state: choices.append((Q[s,a],nxt))
    if not choices: break
    nxt=max(choices,key=lambda z:z[0])[1]
    if nxt in path: break
    state=nxt; path.append(state)
    if state==goal: break
print("Learned path:", path)
print("Goal reached:", state==goal)
display = maze.copy().astype(float)
for r,c in path: display[r,c]=2
plt.imshow(display, cmap="viridis")
plt.title("Maze: 0=Free, 1=Wall, 2=Agent Path"); plt.colorbar(); plt.show()
