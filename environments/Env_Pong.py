import numpy as np
import cv2
import random


'''
     This is the Pong environment. Pong is a classic video game that simulates table tennis, where two players hit a ball
     back and forth across a virtual table using paddles. The goal of the game is to score points by hitting the ball 
     past the opponent's paddle without the opponent returning it.

     In this environment, you will be controlling both paddles (on the left and one the right) which can both move
     up, move down, or remain where they are, resulting in 9 different combinations of actions that can be taken
     
     Author : Martijn Folmer
     Date : 23-04-2023
'''

class Environment:
    def __init__(self):
        """
        Initialize the Pong environment, which is two paddles playing a game of tennis
        """

        self.field_size = 640, 480

        # Paddles
        self.paddle_size = 10, 75     # size of the paddle to draw
        self.paddle_rectangle = [-self.paddle_size[0]/2, -self.paddle_size[1]/2, self.paddle_size[0]/2, self.paddle_size[1]/2]
        self.paddle_xloc = 20       # how far of the edge do we put the paddle
        self.paddle_speed = 10       # how fast the paddle can move
        self.paddle_lim = [50, 430]    # how far the paddle can move up or down

        # Ball
        self.ball_size = 20, 20
        self.ball_rectangle = [-self.ball_size[0]/2*1.5, -self.ball_size[1]/2*1.5, self.ball_size[0]/2*1.5, self.ball_size[1]/2*1.5]

        self.tau = 2  # Duration of each time step

        # Actions we can take
        self.actions = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                self.actions.append([i,j])
        self.actions_n = len(self.actions)  # Number of available actions

        # Threshold values for episode termination
        self.field_threshold = [0, 1]       # how far the ball must be for us to say that it is off
        self.time_step = 0                  # timer
        self.time_max = 1000                # how long a single episode can last

        self.num_hit = 0            # how many times the ball has hit a paddle
        self.got_bounce = 0         # if the ball has hit a paddle in the last frame

        # For visualisation
        self.render_size = self.field_size[0], self.field_size[1]  # The current size of what we show with the render
        self.createRenderBG()  # create the background once

        # Initialize state
        self.reset()

    def reset(self):
        """
        Reset the environment to an initial state and return the state
        """
        self.time_step = 0
        self.num_hit = 0
        self.state = [0.5, 0.5, 0.5, 0.5, random.choice([-5, 5])/10, (random.random() * 10 - 5)/10]

        return self.state

    def calculateNextState(self, state, action):
        """
        Calculates the next state given an action
        :param state: The current state of the simulation, [y_left_paddle, y_right_paddle, x_ball, y_ball, ball_v_x_speed, ball_v_y_speed]
        :param action: The current action, either 0 or 1, representing a negative or positive force on the cart
        :return: newState and whether the simulation is done
        """
        self.got_bounce = 0
        done = False

        paddleL_y, paddleR_y, ball_x, ball_y, ball_v_x, ball_v_y = state

        # update the paddles
        paddleL_y = paddleL_y * self.field_size[1]
        paddleR_y = paddleR_y * self.field_size[1]

        paddleL_v = self.actions[action][0] * self.paddle_speed * self.tau
        paddleR_v = self.actions[action][1] * self.paddle_speed * self.tau

        paddleL_y = min(max(self.paddle_lim[0], paddleL_y + paddleL_v), self.paddle_lim[1])
        paddleR_y = min(max(self.paddle_lim[0], paddleR_y + paddleR_v), self.paddle_lim[1])

        # update the ball
        ball_x = ball_x * self.field_size[0]
        ball_y = ball_y * self.field_size[1]

        ball_x += ball_v_x * 10 * self.tau
        ball_y += ball_v_y * 10 * self.tau

        # check collision with outside
        if ball_x>self.field_size[0] or ball_x<0:
            done = True

        # if we are not done, check collisions
        if not done:

            # bounce with top and bottom of the field
            if ball_y<0 or ball_y>self.field_size[1]:
                ball_v_y *= -1

            # Check ball collisions with paddles (which shoots them back and increases speed)
            paddleL_x = self.paddle_xloc
            paddleR_x = self.field_size[0] - self.paddle_xloc
            paddleL = [int(paddleL_x + self.paddle_rectangle[0]), int(paddleL_y + self.paddle_rectangle[1]),int(paddleL_x + self.paddle_rectangle[2]), int(paddleL_y + self.paddle_rectangle[3])]
            paddleR = [int(paddleR_x + self.paddle_rectangle[0]), int(paddleR_y + self.paddle_rectangle[1]), int(paddleR_x + self.paddle_rectangle[2]), int(paddleR_y + self.paddle_rectangle[3])]

            ball = [int(ball_x + self.ball_rectangle[0]), int(ball_y + self.ball_rectangle[1]), int(ball_x + self.ball_rectangle[2]), int(ball_y + self.ball_rectangle[3])]

            if ball_x > paddleL[2] and self.check_overlap(paddleL, ball):
                ball_v_x *= -1.1
                self.got_bounce = 1
            elif ball_x < paddleR[0] and self.check_overlap(paddleR, ball):
                ball_v_x *= -1.1
                self.got_bounce = 1

        # Return to state
        state = [paddleL_y/self.field_size[1], paddleR_y/self.field_size[1], ball_x/self.field_size[0], ball_y/self.field_size[1], ball_v_x, ball_v_y]

        return state, done

    def calculateReward(self, state):
        """
        Calculates the reward for a given state, which is a numerical value representing how good the current situation
        is

        if the ball has just bounced of a paddle, the reward is 200
        if the ball is in the field, the reward is proportional to the y-distance between it and the closest paddle,
        times a 100
        Else if the ball is out of bound (it has been scored), the score is -100

        :param state: [y_left_paddle, y_right_paddle, x_ball, y_ball, ball_v_x_speed, ball_v_y_speed]
        :return: reward: an int representing how good the current situation is
        """

        if state[2] < 0.5:
            ycheck = state[0]
        else:
            ycheck = state[1]
        reward = (1 - abs(state[3] - ycheck)) * 100

        if self.got_bounce != 0:
            return 200
        elif self.time_step >= self.time_max:
            return -100
        else:
            return reward

    def step(self, action):
        """
        Performs a single timestep on the environment given an action.

        :param action: (list) The current state of the environment,  [y_left_paddle, y_right_paddle, x_ball, y_ball, ball_v_x_speed, ball_v_y_speed]
        :return: the updated state (list), the reward for reaching that state, whether the system is done and an info variable
        """

        # Increase our timer
        self.time_step += 1

        # calculate the next state given the action we wish to perform
        self.state, done = self.calculateNextState(self.state.copy(), action)
        self.state = np.array(self.state)

        # Assign reward
        reward = self.calculateReward(self.state)

        # check if we are past the maximum time allotted for this simulation step
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
        cv2.putText(bg, "Pong", (240, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.bg = bg  # create the background of our render
        pass


    def render(self, state, action = 0):
        """
        Render the environment with a given state and action

        :param state: Current state of the environment [y_left_paddle, y_right_paddle, x_ball, y_ball, ball_v_x_speed, ball_v_y_speed]
        :param action: between 0 and 8, representing how we move the paddles. Not used in current render
        :return: An image showing the current state of the environment
        """

        bg = self.bg.copy()

        # draw the left and right paddles
        paddleL_y = state[0] * self.field_size[1]
        paddleR_y = state[1] * self.field_size[1]
        paddleL_x = self.paddle_xloc
        paddleR_x = self.field_size[0] - self.paddle_xloc

        bg = cv2.rectangle(bg, (int(paddleL_x + self.paddle_rectangle[0]), int(paddleL_y + self.paddle_rectangle[1])),
                           (int(paddleL_x + self.paddle_rectangle[2]), int(paddleL_y + self.paddle_rectangle[3])),
                           (255, 255, 255), -1)
        bg = cv2.rectangle(bg, (int(paddleR_x + self.paddle_rectangle[0]), int(paddleR_y + self.paddle_rectangle[1])),
                           (int(paddleR_x + self.paddle_rectangle[2]), int(paddleR_y + self.paddle_rectangle[3])),
                           (255, 255, 255), -1)

        # draw the ball
        ball_x = state[2] * self.field_size[0]
        ball_y = state[3] * self.field_size[1]
        bg = cv2.rectangle(bg, (int(ball_x + self.ball_rectangle[0]), int(ball_y + self.ball_rectangle[1])),
                           (int(ball_x + self.ball_rectangle[2]), int(ball_y + self.ball_rectangle[3])),
                           (255, 255, 255), -1)

        bg = cv2.putText(bg, f"TimeStep : {self.time_step}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return bg

    # Miscellaneous functions
    def check_overlap(self, rect1, rect2):
        """
        check if any of the two rectangles overlap or not
        :param rect1: given by its corner coordinates [x1, y1, x2, y2]
        :param rect2: given by its corner coordinates [x3, y3, x4, y4]
        :return: boolean, True for overlap, False for not overlapping
        """

        # Get the coordinates of the intersection rectangle
        x_overlap = max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
        y_overlap = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))

        # Check if there is any overlap
        if x_overlap * y_overlap > 0:
            return True
        else:
            return False


# Show our visualisation
if __name__ == "__main__":
    P = Environment()

    action = 0  # means moving up for both paddles
    while True:

        state, reward, done, _ = P.step(action)
        print(state)
        print(reward)
        print(done)

        img = P.render(P.state, action)
        cv2.imshow('Pong', img)
        cv2.waitKey(-1)

