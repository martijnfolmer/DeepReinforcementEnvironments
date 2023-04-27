import numpy as np
import cv2
import math

'''
    This is the CartPole environment. It consists of a cart on a line, with a inverted pendulum pointing upwards. The
    goal is to keep this pendulum upright by applying forces to the cart. You can either apply a force to the left or
    a force to the right. Applying no force is not an option

    The simulation is over when the maximum time allotted has been reached, the cart goes to far along the line or the
    inverted pendulum tips over too much.

    Author : Martijn Folmer
    Date : 23-04-2023
'''

class Environment:
    def __init__(self):
        """
        Initialize the CartPole environment, which is a inverted pendulum system
        """

        # Physical properties
        self.gravity = 9.8                                      # Gravity
        self.masscart = 1.0                                     # Mass of the cart
        self.masspole = 0.1                                     # mass of the pole
        self.total_mass = (self.masspole + self.masscart)       # total mass of entire cart
        self.length = 0.5                                       # Length of the pole
        self.polemass_length = (self.masspole * self.length)    # Mass * length
        self.force_mag = 10.0                                   # the amount of force we exert on the cart with a left or right push
        self.tau = 0.05                                         # Duration of each time step

        # Actions we can take
        self.actions = [-self.force_mag, self.force_mag]  # either apply a negative force, or a positive force
        self.actions_n = len(self.actions)  # Number of available actions

        # Threshold values for episode termination
        self.x_threshold = 2.4                                  # if x exceeds +- threshold, we are done
        self.theta_threshold_radians = 12 * 2 * np.pi / 360     # if theta exceeds this amount of radians
        self.time_step = 0                                      # timer
        self.time_max = 200                                     # how long a single episode can last

        # For visualisation
        self.render_size = 640, 480  # The current size of what we show with the render
        self.createRenderBG()  # create the background once

        # Initialize state
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state and return the state
        """
        self.time_step = 0
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))  # [xposition, xdot, angle, angledot]
        return self.state

    def calculateNextState(self, state, action):
        """
        Calculates the next state given an action
        :param state: The current state of the simulation, [x, xdot, angle of pole, angledot of pole]
        :param action: The current action, either 0 or 1, representing a negative or positive force on the cart
        :return: newState and whether the simulation is done
        """

        x, x_dot, theta, theta_dot = state

        # Calculate force and acceleration
        force = self.actions[action]  # the horizontal force with which we push  on the cart
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # The physics calculations for angle acceleration and cart acceleration
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass  # Force due to agents action
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))  # acceleration of the pole
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass  # acceleration of the cart

        # Update state
        x = x + self.tau * x_dot                        # change x location
        x_dot = x_dot + self.tau * xacc                 # change cart speed
        theta = theta + self.tau * theta_dot            # change angle
        theta_dot = theta_dot + self.tau * thetaacc     # change angular speed

        # Calculate whether we are done, which is when we are of the side, or the pendulum rotates too much
        done = (x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

        return [x, x_dot, theta, theta_dot], done

    def calculateReward(self, state):
        """
        Calculates the reward for a given state, which is a numerical value representing how good the current situation
        is

        :param state: the current state of the environment [x, xdot, xgoal]
        :return: reward: an int representing how good the current situation is
        """
        x, x_dot, theta, theta_dot = state

        done = (x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians)

        # Reward is 1 if we are good, reward is 0 if we are done
        reward = 1.0 if not done else 0.0

        return reward

    def step(self, action):

        """
        Performs a single timestep on the environment given an action.

        :param action: (list) The current state of the environment, [x, xdot, xgoal]
        :return: the updated state (list), the reward for reaching that state, whether the system is done and an info variable
        """

        # Increase time step
        self.time_step += 1

        # Calculate the next state in the environment
        self.state, done = self.calculateNextState(self.state.copy(), action)
        self.state = np.array(self.state)

        # Assign reward
        reward = self.calculateReward(self.state.copy())

        # check if we are past our best time step
        if self.time_step >= self.time_max:
            done = True

        # Return state, reward, and termination status
        return self.state, reward, done, {}


    def createRenderBG(self):
        """
        Create the background for the render step once, this saves on computation
        :return: -- (set self.bg as background)
        """
        bg = np.zeros((self.render_size[1], self.render_size[0], 3))
        bg = cv2.line(bg, (40, 250), (560, 250), (255, 255, 255), 9)
        cv2.putText(bg, "CartPole environment", (170, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.bg = bg        # create the background of our render


    def render(self, state, action):
        """
        Render the environment with a given state and action

        :param state: Current state of the environment [x, xdot, xgoal]
        :param action: 0 or 1, representing a force to the left or a force to the right
        :return: An image showing the current state of the environment
        """

        x, _, theta, _ = state

        # Get the background we draw our state on top of
        bg = self.bg.copy()

        # draw cart
        xcart = (x - -self.x_threshold)/(self.x_threshold * 2) * self.render_size[0]

        cartwidth, cartheight = 100, 50
        x1, y1 = int(xcart-cartwidth/2), int(self.render_size[1]/2-cartheight/2)
        x2, y2 = int(xcart+cartwidth/2), int(self.render_size[1]/2+cartheight/2)
        bg = cv2.rectangle(bg, (x1, y1), (x2, y2), (255, 0, 0), -1)
        bg = cv2.circle(bg, (int(xcart), int(self.render_size[1]/2)), 10, (0, 255, 0), -1)

        # draw Pole
        pixelSize_pole = self.length/(2*self.x_threshold) * self.render_size[0]
        x1, y1 = int(xcart), int(self.render_size[1]/2)
        x2, y2 = int(x1 + math.cos(theta - math.pi/2)*pixelSize_pole), int(y1 + math.sin(theta - math.pi/2)*pixelSize_pole)
        bg = cv2.line(bg, (x1, y1), (x2, y2), (0, 0, 255), 5)

        # draw action
        if action == 0:
            cv2.arrowedLine(bg, (280, 400), (220, 400), (0, 0, 255), 5, tipLength=0.5)
        else:
            cv2.arrowedLine(bg, (360, 400), (420, 400), (0, 255, 0), 5, tipLength=0.5)
        cv2.putText(bg, "F", (305, 425), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        return bg


# Show our visualisation
if __name__=="__main__":
    CP = Environment()

    # perform steps, see what happens
    action = 0 # force to the left
    while True:
        state, reward, done, _ = CP.step(action)
        # action = abs(action-1)
        img = CP.render(state, action)
        cv2.imshow('render', img)
        cv2.waitKey(-1)
