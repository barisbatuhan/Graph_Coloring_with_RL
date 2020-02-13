# valueIterationAgents.py
# -----------------------
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


import mdp, util, random

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        vcurr = util.Counter()
        for iter in range(self.iterations):
            vcurr = self.values.copy()
            all_states = self.mdp.getStates()
            for state in all_states:
                if(self.mdp.isTerminal(state)):
                    self.values[state] = 0
                else:
                    action = self.computeActionFromValues(state)
                    actions = self.mdp.getPossibleActions(state)
                    best_val = float("-inf")
                    for action in actions:
                        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                        n_val = 0
                        for trans in transitions:
                            n_val += trans[1] * (self.mdp.getReward(state, action, trans[0]) + self.discount * vcurr[trans[0]])
                        if(n_val > best_val):
                            best_val = n_val
                    self.values[state] = best_val

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getNextState(self, state, action):
        next_state = [0, 0]
        if(action == 'west'):
            next_state[0] = state[0] - 1
            next_state[1] = state[1]
        elif(action == 'east'):
            next_state[0] = state[0] + 1
            next_state[1] = state[1]
        elif(action == 'north'):
            next_state[0] = state[0] 
            next_state[1] = state[1] + 1
        elif(action == 'south'):
            next_state[0] = state[0] 
            next_state[1] = state[1] - 1
        else:
            next_state = str('TERMINAL STATE')

        return next_state

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        n_val = 0
        for trans in transitions:
            n_val += trans[1] * (self.mdp.getReward(state, action, trans[0]) + self.discount * self.values[trans[0]])
        return n_val

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if(self.mdp.isTerminal(state)):
            return None 
        
        actions = self.mdp.getPossibleActions(state)
        best_val = float("-inf")
        best_action = None
        for action in actions:
            n_val = self.computeQValueFromValues(state, action)
            if(n_val > best_val):
                best_action = action
                best_val = n_val
        return best_action    
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
