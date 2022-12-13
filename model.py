#import the game
import gym_super_mario_bros
#impor the joypad wrapper
from nes_py.wrappers import JoypadSpace
#import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO 
# import stable_baselines3
# import PPO from stable_baselines3

env=gym_super_mario_bros.make('SuperMarioBros-v0')
env=JoypadSpace(env,SIMPLE_MOVEMENT)

# model = PPO.load('C:\\Users\\navin\\Downloads\\best_model_1000000')
model = PPO.load('./train/best_model_950000')
# print(model)
state = env.reset()
# # Start the game 
# # Loop through the game
done=False
while not done: 
    
    action, _ = model.predict(state.copy())
    # action, _ = model.predict(state)
    # temp = action.copy()
    # print(temp)
    # print(int(action))
    state, reward, done, info = env.step(int(action))
    env.render()