
"""
@author: Ju Shen
@email: jshen1@udayton.edu
@date: 02-16-2023
"""
import math
import random
import numpy as np
import math as mth
from tqdm import tqdm

# The state class
class State:
    def __init__(self, angle1=0, angle2=0):
        self.angle1 = angle1
        self.angle2 = angle2

class ReinforceLearning:

    #
    def __init__(self, unit=5):

        ####################################  Needed: here are the variable to use  ################################################

        # The crawler agent
        self.crawler = 0

        # Number of iterations for learning
        self.steps = 1000

        # learning rate alpha
        self.alpha = 0.2

        # Discounting factor
        self.gamma = 0.95

        # E-greedy probability
        self.epsilon = 0.1

        self.Qvalue = []  # Update Q values here
        self.unit = unit  # 5-degrees
        self.angle1_range = [-35, 55]  # specify the range of "angle1"
        self.angle2_range = [0, 180]  # specify the range of "angle2"
        self.rows = int(1 + (self.angle1_range[1] - self.angle1_range[0]) / unit)  # the number of possible angle 1
        self.cols = int(1 + (self.angle2_range[1] - self.angle2_range[0]) / unit)  # the number of possible angle 2

        ########################################################  End of Needed  ################################################



        self.pi = [] # store policies
        self.actions = [-1, +1] # possible actions for each angle

        # Controlling Process
        self.learned = False



        # Initialize all the Q-values
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.Qvalue.append(row)



        # Initialize all the action combinations
        self.actions = ((-1, -1), (-1, 0), (0, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


        # Initialize the policy
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(random.randint(0, 8))
            self.pi.append(row)





    # Reset the learner to empty
    def reset(self):
        self.Qvalue = [] # store Q values
        self.R = [] # not working
        self.pi = [] # store policies

        # Initialize all the Q-values
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.Qvalue.append(row)

        # Initiliaize all the Reward
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            self.R.append(row)

        # Initialize all the action combinations
        self.actions = ((-1, -1), (-1, 0), (0, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))


        # Initialize the policy
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(random.randint(0, 8))
            self.pi.append(row)

        # Controlling Process
        self.learned = False

    # Set the basic info about the robot
    def setBot(self, crawler):
        self.crawler = crawler


    def storeCurrentStatus(self):
        self.old_location = self.crawler.location
        self.old_angle1 = self.crawler.angle1
        self.old_angle2 = self.crawler.angle2
        self.old_contact = self.crawler.contact
        self.old_contact_pt = self.crawler.contact_pt
        self.old_location = self.crawler.location
        self.old_p1 = self.crawler.p1
        self.old_p2 = self.crawler.p2
        self.old_p3 = self.crawler.p3
        self.old_p4 = self.crawler.p4
        self.old_p5 = self.crawler.p5
        self.old_p6 = self.crawler.p6

    def reverseStatus(self):
        self.crawler.location = self.old_location
        self.crawler.angle1 = self.old_angle1
        self.crawler.angle2 = self.old_angle2
        self.crawler.contact = self.old_contact
        self.crawler.contact_pt = self.old_contact_pt
        self.crawler.location = self.old_location
        self.crawler.p1 = self.old_p1
        self.crawler.p2 = self.old_p2
        self.crawler.p3 = self.old_p3
        self.crawler.p4 = self.old_p4
        self.crawler.p5 = self.old_p5
        self.crawler.p6 = self.old_p6



    def updatePolicy(self):
        # After convergence, generate policy y
        for r in range(self.rows):
            for c in range(self.cols):
                max_idx = 0
                max_value = -1000
                for i in range(9):
                    if self.Qvalue[r][9 * c + i] >= max_value:
                        max_value = self.Qvalue[r][9 * c + i]
                        max_idx = i

                # Assign the best action
                self.pi[r][c] = max_idx


    # This function will do additional saving of current states than Q-learning
    def onLearningProxy(self, option):
        self.storeCurrentStatus()
        if option == 0:
            self.onMonteCarlo()
        elif option == 1:
            self.onTDLearning()
        elif option == 2:
            self.onQLearning()
        self.reverseStatus()


        # Turn off learned
        self.learned = True



    # For the play_btn uses: choose an action based on the policy pi
    def onPlay(self, ang1, ang2, mode=1):

        epsilon = self.epsilon

        ang1_cur = ang1
        ang2_cur = ang2

        # get the state index
        r = int((ang1_cur - self.angle1_range[0]) / self.unit)
        c = int((ang2_cur - self.angle2_range[0]) / self.unit)

        # Choose an action and udpate the angles
        idx, angle1_update, angle2_update = self.chooseAction(r=r, c=c)
        ang1_cur += self.unit * angle1_update
        ang2_cur += self.unit * angle2_update

        return ang1_cur, ang2_cur



    ####################################  Needed: here are the functions you need to use  ################################################


    # This function is similar to the "runReward()" function but without returning a reward.
    # It only update the robot position with the new input "angle1" and "angle2"
    def setBotAngles(self, ang1, ang2):
        self.crawler.angle1 = ang1
        self.crawler.angle2 = ang2
        self.crawler.posConfig()



    # Given the current state, return an action index and angle1_update, angle2_update
    # Return valuse
    #  - index: any number from 0 to 8, which indicates the next action to take, according to the e-greedy algorithm
    #  - angle1_update: return the angle1 new value according to the action index, one of -1, 0, +1
    #  - angle2_update: the same as angle1

    def chooseAction(self, r, c):
        # implementation here
        angles = [(-1, -1), (-1, 0), (-1,1), (0, -1), (0,0), (0, 1), (1, -1), (1, 0), (1, 1)]

        max_val = -100
        max_idx = 0

        if np.random.uniform(0,1) < self.epsilon:
            max_idx = np.random.randint(0,9)
        else:
            for i in range(9):
                if self.Qvalue[r][c*9+i] > max_val:
                    max_val = self.Qvalue[r][c*9+i]
                    max_idx = i

        angle1_update,angle2_update = angles[max_idx]

        # below is just an example of randomly generating angle updates
        # idx = 0
        # angle1_update = random.randint(-1, 1)
        # angle2_update = random.randint(-1, 1)

        # if out of the range, then just make angle1_update = 0
        # if angle1_update * self.unit + self.crawler.angle1 < self.angle1_range[0] or angle1_update * self.unit + self.crawler.angle1 >  self.angle1_range[1]:
        #     angle1_update = 0
        if (angle1_update * self.unit) + self.crawler.angle1 not in [*range(-35, 60, 5)]:
            angle1_update = 0

        # if out of the range, then just make angle2_update = 0
        # if angle2_update * self.unit + self.crawler.angle2 < self.angle2_range[0] or angle2_update * self.unit + self.crawler.angle2 > self.angle2_range[1]:
        #     angle2_update = 0
        if (angle2_update * self.unit) + self.crawler.angle2 not in [*range(0,185,5)]:
            angle2_update = 0


        return max_idx, angle1_update, angle2_update

    def calculate_reward(self,pow_count,state,action,episode):
        if episode[1] != 'Exit':
            if state.angle1 == episode[0].angle1 and state.angle2 == episode[0].angle2 and episode[1] == action:
                return self.calculate_reward(pow_count,state,action,episode[3:])
            else:
                return pow(self.gamma,pow_count) * episode[2] + self.calculate_reward(pow_count+1,state,action,episode[3:])
        else:
            return pow(self.gamma,pow_count) * episode[2]

    # Method 1: Monte Carlo algorithm
    def onMonteCarlo(self):
        # You need to implement this function for the project 4 part 1
        state_action_reward_list = []
        no_of_t_state_action = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            state_action_reward_list.append(row)

        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                for a in range(9):
                    row.append(0.0)
            no_of_t_state_action.append(row)
        # cur_loc = self.crawler.location
        rows = [*range(-35, 60, 5)]
        cols = [*range(0, 185, 5)]
        angles = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        # self.setBotAngles(ang1,ang2)
        episodes = []

        '''setup for-loop to generate episodes'''
        '''Generate trajectory'''
        '''Policy Evaluation'''
        for _ in range(self.steps):
            org_loc = self.crawler.location
            episode = []
            ang1 = random.choice(rows)
            ang2 = random.choice(cols)
            self.setBotAngles(ang1, ang2)
            episode.append(State(ang1, ang2))
            for t in range(100):
                action = self.pi[rows.index(ang1)][cols.index(ang2)]
                episode.append(action)
                episode.append(float(0))
                ang1, ang2 = ang1 + (angles[action][0] * 5), ang2 + (angles[action][1] * 5)
                if ang1 * self.unit not in [*range(-35, 60, 5)]:
                    ang1 = 0
                if ang2 * self.unit not in [*range(0, 185, 5)]:
                    ang2 = 0
                episode.append(State(ang1,ang2))
                self.setBotAngles(ang1,ang2)

            terminated_loc = self.crawler.location
            utility = terminated_loc[0] - org_loc[0]
            episode.append('Exit')
            episode.append(utility)
            episodes.append(episode)
        for each_episode in episodes:
            for k in range(0,len(each_episode),3):
                reward = self.calculate_reward(0,each_episode[k],each_episode[k+1],each_episode[k:])
                try:
                    state_action_reward_list[rows.index(each_episode[k].angle1)][(cols.index(each_episode[k].angle2) * 9) + (each_episode[k+1]) ] = state_action_reward_list[rows.index(each_episode[k].angle1)][(cols.index(each_episode[k].angle2) * 9) + (each_episode[k+1]) ] + reward
                except:
                    '''Handling exit word in episode'''
                    pass
                try:
                    no_of_t_state_action[rows.index(each_episode[k].angle1)][(cols.index(each_episode[k].angle2) * 9) + (each_episode[k+1]) ] += 1
                except:
                    '''Handling exit word in episode'''
                    pass
        for i in range(0,len(self.Qvalue)):
            for j in range(0,len(self.Qvalue[i])):
                try:
                    self.Qvalue[i][j] = state_action_reward_list[i][j] / no_of_t_state_action[i][j]
                except Exception as e:
                    '''Handling zero by zero division error'''
                    pass
        self.updatePolicy()
        return


    def round_off(self):
        for i in range(len(self.Qvalue)):
            for j in range(len(self.Qvalue[i])):
                temp = round(self.Qvalue[i][j],2)
                self.Qvalue[i][j] = temp
    # Method 2: Temporal Difference based on SARSA
    def onTDLearning(self):
        rows = [*range(-35, 60, 5)]
        cols = [*range(0, 185, 5)]

        for _ in tqdm(range(self.steps),desc = 'TD-SARSA'):
            epi_start_loc = self.crawler.location
            episode = []
            ang1 = self.crawler.angle1
            ang2 = self.crawler.angle2
            self.setBotAngles(ang1, ang2)
            episode.append(State(ang1, ang2))
            for t in range(100):
                loc_before_action = self.crawler.location
                action,ang1_update,ang2_update = self.chooseAction(rows.index(ang1), cols.index(ang2))
                ang1, ang2 = ang1 + (ang1_update * 5), ang2 + (ang2_update * 5)
                self.setBotAngles(ang1, ang2)
                loc_after_action = self.crawler.location
                immediate_reward = loc_after_action[0] - loc_before_action[0]
                if immediate_reward != 0 or immediate_reward != 0.0:
                    pass
                episode.append(action)
                episode.append(immediate_reward)
                episode.append(State(ang1, ang2))

            epi_end_loc = self.crawler.location
            epi_reward = epi_end_loc[0] - epi_start_loc[0]
            episode.append('Exit')
            episode.append(epi_reward)
            '''Rewriting the above formula
                            Q(s,a) = Q(s,a) + alpha[R + discount * Q(S',A') - Q(s,a)]
                            Q(s,a) = [(1-self.alpha) * Q(s,a)] + self.alpha[R + discount * Q(S',A')]
                            '''
            for i in range(0,len(episode),3):
                '''
                converting formula into steps
                a = reward + discount * Q(S',A')
                b = self.alpha * [a - Q(s,a)]
                Q(s,a) += b
                '''
                try:
                    state_row = rows.index(episode[i].angle1)
                    state_col = cols.index(episode[i].angle2)
                    state_action = episode[i+1]
                    state_prime_row = rows.index(episode[i+3].angle1)
                    state_prime_col = cols.index(episode[i+3].angle2)
                    state_prime_action = episode[i+4]
                    a = episode[i+2] + (self.gamma * self.Qvalue[state_prime_row][(state_prime_col*9)+state_prime_action])
                    if math.isnan(a):
                        pass
                    b = self.alpha * (a - self.Qvalue[state_row][(state_col * 9)+state_action])
                    if math.isnan(b):
                        pass
                    self.Qvalue[state_row][(state_col * 9) + state_action] += b
                    if math.isnan(self.Qvalue[state_row][(state_col * 9)+state_action]):
                        pass
                except:
                    '''Handling episode Exiting rather than decrasing episode length'''
        return




    # Method 3: Bellman operator based Q-learning
    def onQLearning(self):
        # You don't have to work on it for the moment
        rows = [*range(-35, 60, 5)]
        cols = [*range(0, 185, 5)]

        for _ in tqdm(range(self.steps), desc='Q-Learning'):
            epi_start_loc = self.crawler.location
            episode = []
            ang1 = self.crawler.angle1
            ang2 = self.crawler.angle2
            self.setBotAngles(ang1, ang2)
            episode.append(State(ang1, ang2))
            for t in range(100):
                loc_before_action = self.crawler.location
                action, ang1_update, ang2_update = self.chooseAction(rows.index(ang1), cols.index(ang2))
                ang1, ang2 = ang1 + (ang1_update * 5), ang2 + (ang2_update * 5)
                self.setBotAngles(ang1, ang2)
                loc_after_action = self.crawler.location
                immediate_reward = loc_after_action[0] - loc_before_action[0]
                if immediate_reward != 0 or immediate_reward != 0.0:
                    pass
                episode.append(action)
                episode.append(immediate_reward)
                episode.append(State(ang1, ang2))

            epi_end_loc = self.crawler.location
            epi_reward = epi_end_loc[0] - epi_start_loc[0]
            episode.append('Exit')
            episode.append(epi_reward)
            for i in range(0, len(episode), 3):
                '''
                converting formula into steps
                a = reward + discount * max(Q(S'))
                b = self.alpha * [a - Q(s,a)]
                Q(s,a) += b
                '''
                try:
                    state_row = rows.index(episode[i].angle1)
                    state_col = cols.index(episode[i].angle2)
                    state_action = episode[i + 1]
                    state_prime_row = rows.index(episode[i + 3].angle1)
                    state_prime_col = cols.index(episode[i + 3].angle2)
                    state_prime_action = episode[i + 4]
                    a = episode[i + 2] + (
                                self.gamma * max(self.Qvalue[state_prime_row][(state_prime_col * 9):(state_prime_col * 9) + 9]))
                    if math.isnan(a):
                        pass
                    b = self.alpha * (a - self.Qvalue[state_row][(state_col * 9) + state_action])
                    if math.isnan(b):
                        pass
                    self.Qvalue[state_row][(state_col * 9) + state_action] += b
                    if math.isnan(self.Qvalue[state_row][(state_col * 9) + state_action]):
                        pass
                except:
                    '''Handling episode Exiting rather than decrasing episode length'''
        return