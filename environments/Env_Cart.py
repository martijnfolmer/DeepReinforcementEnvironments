import numpy as np
import cv2
import random

'''
    This is the Cart environment. It consists of a cart on a line, with a goal of moving to another point on that line.
    The cart can be influenced by either applying a positive force on it, or a negative force on it. Applying no force 
    is not an option.
    
    The simulation is over when the maximum time allotted has been reached, or the cart goes to far along the line.

    Author : Martijn Folmer
    Date : 23-04-2023
'''

class Environment:
    def __init__(self):
        print("Initialized the Cart environment")
        # The actions that can be taken during the step event
        self.actions = [-10, 10]  # Either -1 degrees, add 0 degrees or add 1 degrees
        self.actions_n = len(self.actions)  # Number of available actions

        # Cart variables
        self.mass = 10      # the weight of the car
        self.x_decc = 0.1     # how fast it deccelerates

        # Environment variables
        self.tau = 0.5     # the correlation for how much time a single timestep is
        self.x_range = [0, 100]  # The range how far we can move to the left or right
        self.time_max = 75  # How long we go maximum

        # Variables for rendering
        self.render_size = 640, 480  # The current size of what we show with the render
        self.createRenderBG()  # create the background once

        self.reset()  # Set the initial state, reset time_step and reset done

    def reset(self):
        """
        Resets the environment and prepares it for the next training episode

        :return: --
        """
        self.done = False       # whether the simulation is finished or not
        self.time_step = 0      # timer
        self.state = [50, 0, random.randint(self.x_range[0], self.x_range[1])] # x, x_dot, x_goal

    def step(self, action):

        """
        Performs a single timestep on the environment given an action.

        :param action: (list) The current state of the environment, [x, xdot, xgoal]
        :return: the updated state (list), the reward for reaching that state, whether the system is done and an info variable
        """

        # Calculate the next state
        self.state, done = self.calculateNextState(self.state, action)

        # Calculate the reward for the achieved state
        reward = self.calculateReward(self.state)

        # increase the timer
        self.time_step += 1

        # Check if the simulation is finished or not
        if self.time_step >= self.time_max or done:
            self.done = True

        # Information variable, can be used for bugfixing and returning additional variable values for example
        info = {}

        return self.state, reward, self.done, info

    def calculateReward(self, state):
        """
        Calculates the reward for a given state, which is a numerical value representing how good the current situation
        is

        :param state: the current state of the environment [x, xdot, xgoal]
        :return: reward: an int representing how good the current situation is
        """
        x, x_dot, x_goal = state

        # figure out where we end up
        x_cur = x
        xdot_cur= x_dot
        while True:
            x_cur += xdot_cur
            xdot_cur -= np.sign(xdot_cur) * self.x_decc
            if abs(xdot_cur) <= self.x_decc:
                break

        if abs(x_cur-x_goal) <= 1:
            reward = 100
        else:
            reward = 50 - abs(x_cur-x_goal)

        if x_cur > self.x_range[1] or x_cur < self.x_range[0]:
            reward -= 100

        return reward

    def calculateNextState(self, state, action):
        """
        Calculates the next state given an action
        :param state: The current state of the simulation, [x, xdot, xgoal]
        :param action: The current action, either 0 or 1, representing a negative or positive force on the cart
        :return: newState and whether the simulation is done
        """
        x, x_dot, x_goal = state  # the current x location, the speed and the goal

        # PHYSICS!!!
        force = self.actions[action]        # either -F, or +F
        x_dotdot = force/self.mass          # calculate acceleration
        x_dot += x_dotdot * self.tau - self.x_decc * np.sign(x_dot) * self.tau        # calculate new speed (with decceleration)
        if abs(x_dot) < self.x_decc*self.tau:
            x_dot = 0
        x += x_dot * self.tau               # calculate new position

        # check if simulation is done because the cart is out of bounds
        if x > self.x_range[1] or x < self.x_range[0]:
            done = True
        else:
            done = False

        # return our state
        return [x, x_dot, x_goal], done

    def createRenderBG(self):
        """
        Create the background for the render step once, this saves on computation
        :return: -- (set self.bg as background)
        """
        bg = np.zeros((self.render_size[1], self.render_size[0], 3))
        bg = cv2.line(bg, (40, 250), (560, 250), (255, 255, 255), 9)
        cv2.putText(bg, "Cart environment", (170, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.bg = bg        # create the background of our render

    def render(self, state, action):
        """
        Render the environment with a given state and action

        :param state: Current state of the environment [x, xdot, xgoal]
        :param action: 0 or 1, representing a force to the left or a force to the right
        :return: An image showing the current state of the environment
        """

        x, _, xgoal = state
        bg = self.bg.copy()

        # draw cart
        xcart = (x - self.x_range[0])/(self.x_range[1]) * 560 + 40
        cartwidth, cartheight = 100, 50
        x1, y1 = int(xcart-cartwidth/2), int(250-cartheight/2)
        x2, y2 = int(xcart+cartwidth/2), int(250+cartheight/2)
        bg = cv2.rectangle(bg, (x1, y1), (x2, y2), (255, 0, 0), -1)
        bg = cv2.circle(bg, (int(xcart), 250), 10, (0, 255, 0), -1)

        # draw xgoal
        xgoal = (xgoal - self.x_range[0])/(self.x_range[1]) * 560 + 40
        bg = cv2.circle(bg, (int(xgoal), 250), 20, (0, 0, 255), 5)
        bg = cv2.circle(bg, (int(xgoal), 250), 4, (0, 0, 255), -1)

        # draw action
        if action == 0:
            cv2.arrowedLine(bg, (280, 400), (220, 400), (0, 0, 255), 5, tipLength=0.5)
        else:
            cv2.arrowedLine(bg, (360, 400), (420, 400), (0, 255, 0), 5, tipLength=0.5)
        cv2.putText(bg, "F", (305, 425), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        return bg


# For testing the render
if __name__ == "__main__":
    env = Environment()

    env.calculateReward([50, 4, 75])

    while True:
        env.step(1)
        img = env.render(env.state, 1)

        print(env.calculateReward(env.state.copy()))

        cv2.imshow('render', img)
        cv2.waitKey(-1)

        if env.done:
            break
