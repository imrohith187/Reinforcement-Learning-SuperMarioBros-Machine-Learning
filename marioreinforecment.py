#import the gamepi
import gym_super_mario_bros
#impor the joypad wrapper
from nes_py.wrappers import JoypadSpace
#import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


#Setup game
env=gym_super_mario_bros.make('SuperMarioBros-v0')
# env=JoypadSpace(env,SIMPLE_MOVEMENT)
# print(SIMPLE_MOVEMENT)
# print(type(env.action_space.sample()))

# Create a flag - restart or not
done = True
# Loop through each frame in the game
for step in range(5000): 
    # Start the game to begin with 
    if done: 
        # Start the gamee
        env.reset()
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # Show the game on the screen
    env.render()
# Close the game
env.close()




