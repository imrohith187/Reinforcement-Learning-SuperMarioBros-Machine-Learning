#import the gamepi
import gym_super_mario_bros
#impor the joypad wrapper
from nes_py.wrappers import JoypadSpace
#import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


# Import os for file path management
import os 

# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

env=gym_super_mario_bros.make('SuperMarioBros-v0')
env=JoypadSpace(env,SIMPLE_MOVEMENT)  

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'
# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=512) 
# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=1000000, callback=callback)
model.save('thisisatestmodel')