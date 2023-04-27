from collections import deque
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras

# Get the static environment
from environments.Env_Temperature_dynamic import Environment

"""
    This is the script which uses deep q learning to solve the dynamic Temperature environment, in which the 
    system must try to raise or lower the temperature to approach a given temperature (which is a random temperature
    between 0 and 100 degrees)
    
    Deep q learning is a method of machine learning in which a simulation is played a great number of times, and its 
    inputs, current state and results are recorded. These recording are then used to train a neural network which
    will be able to predict what the results will be for a given action and state.
    
    Running this script will train a model on the dynamic temperature environment and save a video every X number of 
    episodes, so you can see how the model improves over time. At the end, a resulting model will be saved and 
    a graph shown of how well the model performed over time.
    
    This version of deep q learning uses the environment to predict the q values over future states instead of using 
    the neural network to do the same, resulting in a "shallow" version of deep q learning that only looks 1 step into
    the future, but is not at risk of overestimating q values.
    You can alter the code in train() if you want the model to do the predictions (or look in the later environments
    such as Cart which do use a the more complete version of deep q-learning)
    
    Author : Martijn Folmer
    Date : 22-04-2023

"""


class DeepAgent:

    def __init__(self):

        self.environment = Environment()                    # Initialize the environment

        # ML model
        self.action_n = self.environment.actions_n           # how many actions there are
        self.state_shape = [1, len(self.environment.state)]  # The input shape of the model
        self.model = self.createModel(self.state_shape, self.action_n)  # Create the neural network we want to train

        # epsilon greedy variables
        self.epsilon = 1                # Epsilon-greedy algorithm in initialized at 1 meaning every step
                                        # is random at the start
        self.max_epsilon = 1            # maximum epsilon (which means random actions 100% of the time)
        self.min_epsilon = 0.01         # maximum epsilon (which means random actions 1% of the time)
        self.decay = 0.005              # epsilon_n_plus_1 = epsilon * (1-decay)

        # Memory
        self.replay_memory = deque(maxlen=500)       # the memory we replay (First in - First out)

        # Neural network training variables
        self.train_episodes = 500       # How many training episodes we have (= a full run of the environment)
        self.train_per_steps = 4        # We train every X number of steps
        self.train_batch_size = 64      # Batch size of our training
        self.future_decay = 0.75        # how much the future is decayed each time we find a new future reward

        # For testing states
        self.test_states = [[0, 80], [25, 50], [75, 13], [100, 1]]     # what the state is for every
        self.video_every_episode = 5            # How often we create a video

        # Where we want to save the videos
        self.path_to_videos = 'videos'
        if not os.path.exists(self.path_to_videos) : os.mkdir(self.path_to_videos)                  # create folder
        [os.remove(self.path_to_videos + f"/{file}") for file in os.listdir(self.path_to_videos)]   # remove old videos

    def getAction(self):
        """
        This returns either a random action (when we want to explore) or the current best valued action (if we want
        to be greedy). Over time, as epsilon degrees, we will be taking more and more greedy actions

        :return: An action to perform (an int between 0 and 2)
        """

        if np.random.rand() <= self.epsilon:
            # Do a random action so we explore
            action = random.randint(0, self.action_n-1)
        else:
            # Do a greedy action, which is the best action based on our current models output with the given state
            action = self.getBestAction()

        return action

    def getBestAction(self):
        """
        We want to run our model and select the action that it thinks is the best
        :return: An action to perform (an int between 0 and 2)
        """
        # Perform the best action according to our current model
        state = self.environment.state.copy()
        state = np.asarray(state)
        state_reshaped = state.reshape([1, state.shape[0]])

        # predict our best action
        predicted = self.model.predict(state_reshaped)

        # Get the best output (which is the output with the highest q value)
        action = np.argmax(predicted)
        return action

    def createModel(self, state_shape, action_num):

        """
        Create the neural network which is the substitute for our q-table
        :param state_shape: The current state of the environment, which is the input of our model
        :param action_num: The amount of actions we can do, which is the size of the output of the model
        :return: The model
        """

        # A simple machine learning model to replace the q table
        model = keras.Sequential()
        model.add(keras.layers.Dense(48, input_shape=state_shape, activation='relu'))
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(action_num, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        model.summary()

        return model

    def train(self):
        """
        Using the recorded states and rewards, we can train a model to try and predict any future reward for a given
        state.
        :return: --
        """

        # if we don't have enough training steps yet, we return
        if len(self.replay_memory) < self.train_batch_size:
            return

        # Get a random memory sample
        mini_batch = random.sample(self.replay_memory, self.train_batch_size)

        # Create training samples from the mini batch
        X = []      # the states (which are the inputs of the model)
        y = []      # the possible rewards (which are the outputs of the model)
        for batch in mini_batch:

            all_q_values = []               # this is the q values, which is the output of our stuff
            for i in range(self.action_n):
                state_cur = batch[0].copy()
                state_pos = self.environment.calculateNextState(state_cur, i)
                reward_pos = self.environment.calculateReward(state_pos)

                # Look 1 step into the future and check for the future rewards
                future_reward = []
                for j in range(self.action_n):
                    state_try = state_pos.copy()
                    state_future = self.environment.calculateNextState(state_try, j)
                    reward_future = self.environment.calculateReward(state_future)
                    future_reward.append(reward_future)
                future_reward_max = self.future_decay * max(future_reward)
                all_q_values.append(reward_pos+future_reward_max)

            # Append to the current Training batch
            X.append([batch[0]])
            y.append([all_q_values])

        # train the model
        self.model.fit(X, y, epochs=10, verbose=0)

    def test(self, episode_num):
        """
        Run our current model as greedily as possible on a test set, so we can test how well it performs over time

        :param episode_num: which episode we are at, which determines whether we make a video or not
        :return: The rewards the environment has earned during training
        """

        total_testing_rewards = 0
        all_images = []
        for test_state in self.test_states:
            # Set the environment to test situation
            done = False
            self.environment.reset()
            self.environment.state[0] = test_state[0]              # set our test state (current temperature and goal)
            self.environment.state[1] = test_state[1]
            timeSteps = 0
            while not done:
                timeSteps += 1
                # determine action
                action = self.getBestAction()  # get an action to do
                # step action
                newstate, reward, done, _ = self.environment.step(action)

                # increase the rewards given
                total_testing_rewards += reward

                # create all render images we want to append to our video
                if episode_num % self.video_every_episode == 0:
                    all_images.append(self.environment.render(self.environment.state, action))

                # check if we are done or not with the current simulation.
                if done:
                    break

        # every N episodes, we create a video of our test model, so we can observe it over time
        if episode_num % self.video_every_episode == 0:
            vid_cod = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
            name = f'videos/episode_{episode_num}.mp4'
            video_size = all_images[0].shape
            video = cv2.VideoWriter(name, vid_cod, 60, (video_size[1], video_size[0]))
            for image in all_images:
                image = image.astype(dtype=np.uint8)  # this one is very needed
                video.write(image)
            cv2.destroyAllWindows()
            video.release()

        return total_testing_rewards

    def RunDeepQLearning(self):
        """
        Perform the deep Q learning
        :return: --
        """
        print("Start the deep q learning")
        timeSteps = 0       # how much steps we have been training

        total_train_rewards = []
        total_test_rewards = []
        for episode in range(self.train_episodes):
            train_reward = 0          # the total training reward for this episode
            self.environment.reset()            # reset our environment

            done = False
            while not done:
                timeSteps += 1

                # get our current state
                oldstate = self.environment.state.copy()

                # determine action
                action = self.getAction()       # get an action to do

                # step action
                newstate, reward, done, _ = self.environment.step(action)

                # increment the total training rewards
                train_reward += reward

                # append to our memory
                self.replay_memory.append([oldstate, action, reward, newstate, done])

                if timeSteps % self.train_per_steps == 0:
                    self.train()

                if done:
                    total_train_rewards.append(train_reward)
                    print(f"Total training rewards for episode {episode} / {self.train_episodes} : {train_reward}")
                    break

            # run our test every episode:
            test_reward = self.test(episode)
            total_test_rewards.append(test_reward)
            print(f"Total reward after testing : {test_reward}")

            # append epsilon
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * episode)
            print(f"Current epsilon : {self.epsilon}")

        # Save the model
        self.model.save('Temperature_dynamic_final_model')

        # visualise the training and show it
        xstep_train = [i for i in range(len(total_train_rewards))]
        xstep_test = [i for i in range(len(total_test_rewards))]

        plt.rcParams["figure.figsize"] = (20, 10)
        fig, axs = plt.subplots(2, 1)
        self.plot_this(axs[0], xstep_train, total_train_rewards, 0, max(total_train_rewards)*1.1,
                       'Train rewards', 'episode', 'Train reward')
        self.plot_this(axs[1], xstep_test, total_test_rewards, 0, max(total_test_rewards)*1.1,
                       'Test rewards', 'episode', 'Test reward')
        plt.savefig('History_of_training_dynamic_temperature.png')

    # Visualisation of training
    def plot_this(self, axs, X, y, ylim_min, ylim_max, title, xlabel, ylabel):
        axs.plot(X, y)
        axs.set_title(title)
        axs.set_ylabel(ylabel)
        axs.set_xlabel(xlabel)
        axs.set_ylim([ylim_min, ylim_max])
        axs.grid()


if __name__ == "__main__":
    DA = DeepAgent()
    DA.RunDeepQLearning()
