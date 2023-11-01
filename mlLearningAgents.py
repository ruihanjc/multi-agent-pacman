# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random


from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
from typing import List

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm
    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        self.self_position = state.getPacmanPosition()
        self.ghost_position = min(state.getGhostPositions(), key=lambda pos: self.distance(self.self_position, pos))
        self.food_position = state.getFood()
        self.positions = (self.self_position, self.ghost_position, self.food_position)

    def distance(self, pac_pos, ghost_pos):
        return abs(pac_pos[0] - ghost_pos[0]) + abs(pac_pos[1] - ghost_pos[1])

    def __eq__(self, other):
        if other is None:
            return False
        return self.positions == other.positions

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        try:
            return hash(self.positions)
        except TypeError as e:
            print(e)
                # hash(state)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.qVals = {}
        self.previousAction = None
        self.previousState = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        return self.qVals.get((state, action), (0.0, 0))[0]


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state
        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        return max([self.qVals[key][0] for key in self.qVals if key[0] == state], default=0.0)


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"

        self.qVals[(state, action)] = (self.getQValue(state,action) + 
                                       self.alpha * 
                                            (reward + 
                                             self.gamma * 
                                             self.maxQValue(nextState) - 
                                             self.getQValue(state,action)), 
                                       self.qVals[(state, action)][1])


    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        if (state, action) in self.qVals:
            self.qVals[(state, action)] = (self.qVals[((state, action))][0], self.qVals[((state, action))][1] + 1)
        else:
            self.qVals[(state, action)] = (0, 1)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        return self.qVals[(state, action)][1]

    def explorationFn(self,
                        utility: float,
                        counts: int) -> float:
            """
            Computes exploration function.
            Return a value based on the counts
            HINT: Do a greed-pick or a least-pick
            Args:
                utility: expected utility for taking some action a in some given state s
                counts: counts for having taken visited
            Returns:
                The exploration value
            """
            util.raiseNotDefined()

    def getExploration(self, state: GameState ,actions: List)-> Directions:
        if self.previousState:
                self.learn(GameStateFeatures(self.previousState), 
                        self.previousAction, 
                        self.computeReward(self.previousState, state), 
                         GameStateFeatures(state))
        return random.choice(actions)

    def getExploitation(self, stateFeature: GameStateFeatures, 
                              actions: List) -> Directions:
        
        action_values = [(action, self.getQValue(stateFeature, action)) for action in actions]

        # get action with max Q-value
        best_action = max(action_values, key=lambda x: x[1])[0]

        return best_action

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning
        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin
        Args:
            state: the current state
        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

    
        stateFeatures = GameStateFeatures(state)

        for action in legal:
            if (state, action) not in self.qVals:
                self.qVals[(state, action)] = (0, 1)

        if random.uniform(0, 1) < self.epsilon:
            action = self.getExploration(state, legal) 
        else: 
            action = self.getExploitation(stateFeatures, legal)            

        

        self.previousState = state
        self.previousAction = action
        self.updateCount(stateFeatures, action)

        # Now pick what action to take.
        return action
        

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.
        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")
        print("final score", state.getScore())

        stateFeatures = GameStateFeatures(state)

        self.learn(GameStateFeatures(self.previousState), self.previousAction, self.computeReward(self.previousState, state), stateFeatures)

        self.previousActions = None
        self.previousStates = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
