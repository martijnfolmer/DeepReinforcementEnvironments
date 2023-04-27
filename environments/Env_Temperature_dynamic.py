import numpy as np
import cv2
import random


"""
    This is the dynamic temperature environment. The goal for this environment is to reach a given temperature 
    (randomly chosen at start between 0 and 100). The environment current goal starts with a random temperature between 
    0 and 100 degrees celcius and can be raised or decreased by 1 degree every timestep. 
    
    Author : Martijn Folmer
    Date : 22-04-2023
    
"""


class Environment:
    def __init__(self):

        # The actions that can be taken during the step event
        self.actions = [-1, 0, 1]           # Either -1 degrees, add 0 degrees or add 1 degrees
        self.actions_n = len(self.actions)  # Number of available actions

        self.temperature_range = [0, 100]   # The range of available temperatures. If we exceed these numbers, done=true
        self.time_max = 100                 # How long we go maximum
        self.reset()                        # Set the initial state, reset time_step and reset done

        # Variables for rendering
        self.render_size = 640, 480         # The current size of what we show with the render
        self.createRenderBG()               # create the background once

    def reset(self):
        """
        Resets the environment and prepares it for the next training episode

        :return: --
        """
        # current state : [current temperature, goal temperature]
        self.state = [random.randint(self.temperature_range[0], self.temperature_range[1]), 
                      random.randint(self.temperature_range[0], self.temperature_range[1])]
        self.time_step = 0                                  # reset the time
        self.done = False

    def step(self, action):
        """
        Performs a single timestep on the environment given an action.

        :param action: (list) The current state of the environment, [current temperature]
        :return: the updated state (list), the reward for reaching that state, whether the system is done and an info variable
        """
        # calculate the next state
        self.state = self.calculateNextState(self.state, action)

        # Calculate reward based on current state
        reward = self.calculateReward(self.state)

        # increase time step
        self.time_step += 1

        # check if our state is of the charts, in which case we are done
        if self.state[0] < self.temperature_range[0] or self.state[0] > self.temperature_range[1]:
            self.done = True
        # check if we have exceeded the timer for this episode, in which case we are also done
        elif self.time_step >= self.time_max:
            self.done = True

        # Information variable, can be used for bugfixing and returning additional variable values for example
        info = {}

        return self.state, reward, self.done, info

    def calculateNextState(self, state, action):
        """
        Calculates what the next state of the environment is, based on the given state and action

        :param state: (list)  The state of the environment
        :param action: (int)  The index of the action we want to perform
        :return: updated state (list), which is the state after performing the action
        """
        temp_change = self.actions[action]
        state[0] = state[0] + temp_change  # change the temperature

        return state

    def calculateReward(self, state):
        """
            Calculates the reward for a given state input. We use this reward for determining if an action is beneficial
            or not

        :param state: (list) [x], where x is the current temperature
        :return: Reward: (int) A point value based on how beneficial the given state is
        """

        if abs(state[0]-state[1]) <= 0:
            reward = 75
        else:
            reward = 50 - abs(state[0]-state[1])
        return reward

    def createRenderBG(self):
        """
        Create the background for the render step once, this saves on computation
        :return: -- (set self.bg as background)
        """
        bg = np.zeros((self.render_size[1], self.render_size[0], 3))

        self.bg_x1, self.bg_y1 = int(bg.shape[1] * 0.45), int(bg.shape[0] * 0.15)
        self.bg_x2, self.bg_y2 = int(bg.shape[1] * 0.55), int(bg.shape[0] * 0.85)
        bg = cv2.rectangle(bg, (self.bg_x1, self.bg_y1), (self.bg_x2, self.bg_y2), (255, 255, 255), 4)
        bg = cv2.putText(bg, f"{self.temperature_range[0]} C", (self.bg_x2 + 10, self.bg_y2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
        bg = cv2.putText(bg, f"{self.temperature_range[1]} C", (self.bg_x2 + 10, self.bg_y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)

        bg = cv2.putText(bg, f"Temperature dynamic environment", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        self.bg = bg

    def render(self, state, action):
        """
        Render the environment with a given state and action

        :param state: Current state of the environment [current temperature, goal temperature]
        :param action: between 0 and 2, representing how much we change our temperature. Not used in current render
        :return: An image showing the current state of the environment
        """

        # get the background
        bg = self.bg.copy()
        
        # horizontal red line of goal
        y_goal = int(self.state[1] / (self.temperature_range[1] - self.temperature_range[0]) * (self.bg_y1 - self.bg_y2) + self.bg_y2)
        bg = cv2.line(bg, (self.bg_x1 - 15, y_goal), (self.bg_x2 + 15, y_goal), (0, 0, 255), 10)
        bg = cv2.putText(bg, str(self.state[1]), (self.bg_x1 - 70, y_goal+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # horizontal green line of current state
        y_state = int(state[0] / (self.temperature_range[1] - self.temperature_range[0]) * (self.bg_y1-self.bg_y2) + self.bg_y2)
        bg = cv2.line(bg, (self.bg_x1-10, y_state), (self.bg_x2+10, y_state), (0, 255, 0), 6)
        bg = cv2.putText(bg, str(state[0]), (self.bg_x2 + 30, y_state + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the action
        bg = cv2.putText(bg, "Action", (self.bg_x2+150, int((self.bg_y1+self.bg_y2)/2) - 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if action == 0:
            bg = cv2.arrowedLine(bg, (self.bg_x2+200, int((self.bg_y1+self.bg_y2)/2)),
                                 (self.bg_x2+200, int((self.bg_y1+self.bg_y2)/2 + 60)), (0, 0, 255),  5, tipLength=0.5)
        elif action == 1:
            bg = cv2.line(bg, (self.bg_x2+175, int((self.bg_y1+self.bg_y2)/2)),
                          (self.bg_x2+225, int((self.bg_y1+self.bg_y2)/2)), (255, 0, 0), 5)
        else:
            bg = cv2.arrowedLine(bg, (self.bg_x2+200, int((self.bg_y1 + self.bg_y2) / 2)),
                                 (self.bg_x2+200, int((self.bg_y1 + self.bg_y2) / 2 - 60)), (0, 255, 0), 5, tipLength=0.5)

        return bg


# Testing the render function
if __name__ == "__main__":
    Env = Environment()
    print("Test the rendering")
    state_cur = [20]
    action = 2

    img = Env.render(state_cur, action)
    cv2.imshow("render", img)
    cv2.waitKey(-1)
