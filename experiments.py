import numpy as np
import collections
from scipy.sparse import dok_matrix

from mdptoolbox import mdp
from time import clock

from hiive.mdptoolbox import mdp as hmdp

import collections
import pandas as pd

if __name__ == "__main__":

    ACTIONS = 9
    STATES = 3**ACTIONS
    PLAYER = 1
    OPPONENT = 2
    WINS = ([1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 0, 0])

    # The valid number of cells belonging to either the player or the opponent:
    # (player, opponent)
    OWNED_CELLS = ((0, 0),
                   (1, 1),
                   (2, 2),
                   (3, 3),
                   (4, 4),
                   (0, 1),
                   (1, 2),
                   (2, 3),
                   (3, 4))

    def convertIndexToTuple(state):
        """"""
        return(tuple(int(x) for x in np.base_repr(state, 3, 9)[-9::]))

    def convertTupleToIndex(state):
        """"""
        return(int("".join(str(x) for x in state), 3))

    def getLegalActions(state):
        """"""
        return(tuple(x for x in range(ACTIONS) if state[x] == 0))

    def getTransitionAndRewardArrays():
        """"""
        P = [dok_matrix((STATES, STATES)) for a in range(ACTIONS)]
        #R = spdok((STATES, ACTIONS))
        R = np.zeros((STATES, ACTIONS))
        # Naive approach, iterate through all possible combinations
        for a in range(ACTIONS):
            for s in range(STATES):
                state = convertIndexToTuple(s)
                if not isValid(state):
                    # There are no defined moves from an invalid state, so
                    # transition probabilities cannot be calculated. However,
                    # P must be a square stochastic matrix, so assign a
                    # probability of one to the invalid state transitioning
                    # back to itself.
                    P[a][s, s] = 1
                    # Reward is 0
                else:
                    s1, p, r = getTransitionProbabilities(state, a)
                    P[a][s, s1] = p
                    R[s, a] = r
            P[a] = P[a].tocsr()
        #R = R.tolil()
        return(P, R)

    def getTransitionProbabilities(state, action):
        """
        Parameters
        ----------
        state : tuple
            The state
        action : int
            The action

        Returns
        -------
        s1, p, r : tuple of two lists and an int
            s1 are the next states, p are the probabilities, and r is the reward

        """
        #assert isValid(state)
        assert 0 <= action < ACTIONS
        if not isLegal(state, action):
            # If the action is illegal, then transition back to the same state but
            # incur a high negative reward
            s1 = [convertTupleToIndex(state)]
            return(s1, [1], -10)
        # Update the state with the action
        state = list(state)
        state[action] = PLAYER
        if isWon(state, PLAYER):
            # If the player's action is a winning move then transition to the
            # winning state and receive a reward of 1.
            s1 = [convertTupleToIndex(state)]
            return(s1, [1], 1)
        elif isDraw(state):
            s1 = [convertTupleToIndex(state)]
            return(s1, [1], 0)
        # Now we search through the opponents moves, and calculate transition
        # probabilities based on maximising the opponents chance of winning..
        s1 = []
        p = []
        legal_a = getLegalActions(state)
        for a in legal_a:
            state[a] = OPPONENT
            # If the opponent is going to win, we assume that the winning move will
            # be chosen:
            if isWon(state, OPPONENT):
                s1 = [convertTupleToIndex(state)]
                return(s1, [1], -1)
            elif isDraw(state):
                s1 = [convertTupleToIndex(state)]
                return(s1, [1], 0)
            # Otherwise we assume the opponent will select a move with uniform
            # probability across potential moves:
            s1.append(convertTupleToIndex(state))
            p.append(1.0 / len(legal_a))
            state[a] = 0
        # During non-terminal play states the reward is 0.
        return(s1, p, 0)

    def getReward(state, action):
        """"""
        if not isLegal(state, action):
            return -100
        state = list(state)
        state[action] = PLAYER
        if isWon(state, PLAYER):
            return 1
        elif isWon(state, OPPONENT):
            return -1
        else:
            return 0

    def isDraw(state):
        """"""
        try:
            state.index(0)
            return False
        except ValueError:
            return True

    def isLegal(state, action):
        """"""
        if state[action] == 0:
            return True
        else:
            return False

    def isWon(state, who):
        """Test if a tic-tac-toe game has been won.

        Assumes that the board is in a legal state.
        Will test if the value 1 is in any winning combination.

        """
        for w in WINS:
            S = sum(1 if (w[k] == 1 and state[k] == who) else 0
                    for k in range(ACTIONS))
            if S == 3:
                # We have a win
                return True
        # There were no wins so return False
        return False

    def isValid(state):
        """"""
        # S1 is the sum of the player's cells
        S1 = sum(1 if x == PLAYER else 0 for x in state)
        # S2 is the sum of the opponent's cells
        S2 = sum(1 if x == OPPONENT else 0 for x in state)
        if (S1, S2) in OWNED_CELLS:
            return True
        else:
            return False

    P, R = getTransitionAndRewardArrays()
    for discount in np.arange(.1, 1, .2):
        ttt = mdp.ValueIteration(P, R, discount)
        ttt.setVerbose()
        start = clock()
        ttt.run()
        elapsed = clock() - start

    for discount in np.arange(.1, 1, .2):
        ttt = mdp.PolicyIteration(P, R, discount)
        ttt.setVerbose()
        start = clock()
        ttt.run()
        elapsed = clock() - start

    for discount in np.arange(.1, 1, .2):
        qlearner_stats = collections.defaultdict(list)
        ttt = hmdp.QLearning(P, R, discount)
        ttt.setVerbose()
        start = clock()
        ttt.run()
        elapsed = clock() - start
        for stats in ttt.run_stats:
            qlearner_stats['state'].append(stats['State'])
            qlearner_stats['action'].append(stats['Action'])
            qlearner_stats['reward'].append(stats['Reward'])
            qlearner_stats['error'].append(stats['Error'])
            qlearner_stats['time'].append(stats['Time'])
            qlearner_stats['alpha'].append(stats['Alpha'])
            qlearner_stats['epsilon'].append(stats['Epsilon'])
            qlearner_stats['max_v'].append(stats['Max V'])
            qlearner_stats['mean_v'].append(stats['Mean V'])
        qlearner_stats_df = pd.DataFrame(qlearner_stats)
        qlearner_stats_df.to_csv(f'{discount}_qlearner_stats_ttt')
    # Optimal fire management of a threatened species example. source: https://gist.github.com/sawcordwell/bccdf42fcc4e024d394b#file-singlepatch-py

    # The number of population abundance classes
    POPULATION_CLASSES = 7
    # The number of years since a fire classes
    FIRE_CLASSES = 13
    # The number of states
    STATES = POPULATION_CLASSES * FIRE_CLASSES
    # The number of actions
    ACTIONS = 2
    ACTION_NOTHING = 0
    ACTION_BURN = 1

    def check_action(x):
        """Check that the action is in the valid range.
        """
        if not (0 <= x < ACTIONS):
            msg = "Invalid action '%s', it should be in {0, 1}." % str(x)
            raise ValueError(msg)

    def check_population_class(x):
        """Check that the population abundance class is in the valid range.
        """
        if not (0 <= x < POPULATION_CLASSES):
            msg = "Invalid population class '%s', it should be in {0, 1, …, %d}." \
                  % (str(x), POPULATION_CLASSES - 1)
            raise ValueError(msg)

    def check_fire_class(x):
        """Check that the time in years since last fire is in the valid range.
        """
        if not (0 <= x < FIRE_CLASSES):
            msg = "Invalid fire class '%s', it should be in {0, 1, …, %d}." % \
                  (str(x), FIRE_CLASSES - 1)
            raise ValueError(msg)

    def check_probability(x, name="probability"):
        """Check that a probability is between 0 and 1.
        """
        if not (0 <= x <= 1):
            msg = "Invalid %s '%s', it must be in [0, 1]." % (name, str(x))
            raise ValueError(msg)

    def get_habitat_suitability(years):
        """The habitat suitability of a patch relatve to the time since last fire.
        The habitat quality is low immediately after a fire, rises rapidly until
        five years after a fire, and declines once the habitat is mature. See
        Figure 2 in Possingham and Tuck (1997) for more details.
        Parameters
        ----------
        years : int
            The time in years since last fire.
        Returns
        -------
        r : float
            The habitat suitability.
        """
        if years < 0:
            msg = "Invalid years '%s', it should be positive." % str(years)
            raise ValueError(msg)
        if years <= 5:
            return 0.2*years
        elif 5 <= years <= 10:
            return -0.1*years + 1.5
        else:
            return 0.5

    def convert_state_to_index(population, fire):
        """Convert state parameters to transition probability matrix index.
        Parameters
        ----------
        population : int
            The population abundance class of the threatened species.
        fire : int
            The time in years since last fire.
        Returns
        -------
        index : int
            The index into the transition probability matrix that corresponds to
            the state parameters.
        """
        check_population_class(population)
        check_fire_class(fire)
        return population*FIRE_CLASSES + fire

    def convert_index_to_state(index):
        """Convert transition probability matrix index to state parameters.
        Parameters
        ----------
        index : int
            The index into the transition probability matrix that corresponds to
            the state parameters.
        Returns
        -------
        population, fire : tuple of int
            ``population``, the population abundance class of the threatened
            species. ``fire``, the time in years since last fire.
        """
        if not (0 <= index < STATES):
            msg = "Invalid index '%s', it should be in {0, 1, …, %d}." % \
                  (str(index), STATES - 1)
            raise ValueError(msg)
        population = index // FIRE_CLASSES
        fire = index % FIRE_CLASSES
        return (population, fire)

    def transition_fire_state(F, a):
        """Transition the years since last fire based on the action taken.
        Parameters
        ----------
        F : int
            The time in years since last fire.
        a : int
            The action undertaken.
        Returns
        -------
        F : int
            The time in years since last fire.
        """
        ## Efect of action on time in years since fire.
        if a == ACTION_NOTHING:
            # Increase the time since the patch has been burned by one year.
            # The years since fire in patch is absorbed into the last class
            if F < FIRE_CLASSES - 1:
                F += 1
        elif a == ACTION_BURN:
            # When the patch is burned set the years since fire to 0.
            F = 0

        return F

    def get_transition_probabilities(s, x, F, a):
        """Calculate the transition probabilities for the given state and action.
        Parameters
        ----------
        s : float
            The class-independent probability of the population staying in its
            current population abundance class.
        x : int
            The population abundance class of the threatened species.
        F : int
            The time in years since last fire.
        a : int
            The action undertaken.
        Returns
        -------
        prob : array
            The transition probabilities as a vector from state (``x``, ``F``) to
            every other state given that action ``a`` is taken.
        """
        # Check that input is in range
        check_probability(s)
        check_population_class(x)
        check_fire_class(F)
        check_action(a)

        # a vector to store the transition probabilities
        prob = np.zeros(STATES)

        # the habitat suitability value
        r = get_habitat_suitability(F)
        F = transition_fire_state(F, a)

        ## Population transitions
        if x == 0:
            # population abundance class stays at 0 (extinct)
            new_state = convert_state_to_index(0, F)
            prob[new_state] = 1
        elif x == POPULATION_CLASSES - 1:
            # Population abundance class either stays at maximum or transitions
            # down
            transition_same = x
            transition_down = x - 1
            # If action 1 is taken, then the patch is burned so the population
            # abundance moves down a class.
            if a == ACTION_BURN:
                transition_same -= 1
                transition_down -= 1
            # transition probability that abundance stays the same
            new_state = convert_state_to_index(transition_same, F)
            prob[new_state] = 1 - (1 - s)*(1 - r)
            # transition probability that abundance goes down
            new_state = convert_state_to_index(transition_down, F)
            prob[new_state] = (1 - s)*(1 - r)
        else:
            # Population abundance class can stay the same, transition up, or
            # transition down.
            transition_same = x
            transition_up = x + 1
            transition_down = x - 1
            # If action 1 is taken, then the patch is burned so the population
            # abundance moves down a class.
            if a == ACTION_BURN:
                transition_same -= 1
                transition_up -= 1
                # Ensure that the abundance class doesn't go to -1
                if transition_down > 0:
                    transition_down -= 1
            # transition probability that abundance stays the same
            new_state = convert_state_to_index(transition_same, F)
            prob[new_state] = s
            # transition probability that abundance goes up
            new_state = convert_state_to_index(transition_up, F)
            prob[new_state] = (1 - s)*r
            # transition probability that abundance goes down
            new_state = convert_state_to_index(transition_down, F)
            # In the case when transition_down = 0 before the effect of an action
            # is applied, then the final state is going to be the same as that for
            # transition_same, so we need to add the probabilities together.
            prob[new_state] += (1 - s)*(1 - r)

        # Make sure that the probabilities sum to one
        assert (prob.sum() - 1) < np.spacing(1)
        return prob

    def get_transition_and_reward_arrays(s):
        """Generate the fire management transition and reward matrices.
        The output arrays from this function are valid input to the mdptoolbox.mdp
        classes.
        Let ``S`` = number of states, and ``A`` = number of actions.
        Parameters
        ----------
        s : float
            The class-independent probability of the population staying in its
            current population abundance class.
        Returns
        -------
        out : tuple
            ``out[0]`` contains the transition probability matrices P and
            ``out[1]`` contains the reward vector R. P is an  ``A`` × ``S`` × ``S``
            numpy array and R is a numpy vector of length ``S``.
        """
        check_probability(s)

        # The transition probability array
        transition = np.zeros((ACTIONS, STATES, STATES))
        # The reward vector
        reward = np.zeros(STATES)
        # Loop over all states
        for idx in range(STATES):
            # Get the state index as inputs to our functions
            x, F = convert_index_to_state(idx)
            # The reward for being in this state is 1 if the population is extant
            if x != 0:
                reward[idx] = 1
            # Loop over all actions
            for a in range(ACTIONS):
                # Assign the transition probabilities for this state, action pair
                transition[a][idx] = get_transition_probabilities(s, x, F, a)

        return (transition, reward)

    def solve_mdp():
        """Solve the problem as a finite horizon Markov decision process.
        The optimal policy at each stage is found using backwards induction.
        Possingham and Tuck report strategies for a 50 year time horizon, so the
        number of stages for the finite horizon algorithm is set to 50. There is no
        discount factor reported, so we set it to 0.96 rather arbitrarily.
        Returns
        -------
        sdp : mdptoolbox.mdp.FiniteHorizon
            The PyMDPtoolbox object that represents a finite horizon MDP. The
            optimal policy for each stage is accessed with mdp.policy, which is a
            numpy array with 50 columns (one for each stage).
        """
        transition, reward = get_transition_and_reward_arrays(0.5)
        sdp = mdp.FiniteHorizon(transition, reward, 0.96, 50)
        sdp.run()
        return sdp
    def print_policy(policy):
        """Print out a policy vector as a table to console
        Let ``S`` = number of states.
        The output is a table that has the population class as rows, and the years
        since a fire as the columns. The items in the table are the optimal action
        for that population class and years since fire combination.
        Parameters
        ----------
        p : array
            ``p`` is a numpy array of length ``S``.
        """
        p = np.array(policy).reshape(POPULATION_CLASSES, FIRE_CLASSES)
        print("    " + " ".join("%2d" % f for f in range(FIRE_CLASSES)))
        print("    " + "---" * FIRE_CLASSES)
        for x in range(POPULATION_CLASSES):
            print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in
                                         range(FIRE_CLASSES)))

    def simulate_transition(s, x, F, a):
        """Simulate a state transition.
        Parameters
        ----------
        s : float
            The class-independent probability of the population staying in its
            current population abundance class.
        x : int
            The population abundance class of the threatened species.
        F : int
            The time in years since last fire.
        a : int
            The action undertaken.
        Returns
        -------
        x, F : int, int
            The new abundance class, x, of the threatened species and the new years
            last fire class, F.
        """
        check_probability(s)
        check_population_class(x)
        check_fire_class(F)
        check_action(a)

        r = get_habitat_suitability(F)
        F = transition_fire_state(F, a)

        if x == POPULATION_CLASSES - 1:
            # pass with probability 1 - (1 - s)*(1 - r)
            if np.random.random() < (1 - s)*(1 - r):
                x -= 1
        elif 0 < x < POPULATION_CLASSES - 1:
            # pass with probability s
            if np.random.random() < 1 - s:
                if np.random.random() < r: # with probability (1 - s)r
                    x += 1
                else: # with probability (1 - s)(1 - r)
                    x -= 1

        # Add the effect of a fire, making sure x doesn't go to -1
        if a == ACTION_BURN and (x > 0):
            x -= 1

        return x, F

    def _run_tests():
        """Run tests on the modules functions.
        """
        assert get_habitat_suitability(0) == 0
        assert get_habitat_suitability(2) == 0.4
        assert get_habitat_suitability(5) == 1
        assert get_habitat_suitability(8) == 0.7
        assert get_habitat_suitability(10) == 0.5
        assert get_habitat_suitability(15) == 0.5
        state = convert_index_to_state(STATES - 1)
        assert state == (POPULATION_CLASSES - 1, FIRE_CLASSES - 1)
        state = convert_index_to_state(STATES - 2)
        assert state == (POPULATION_CLASSES -1, FIRE_CLASSES - 2)
        assert convert_index_to_state(0) == (0, 0)
        for idx in range(STATES):
            state1, state2 = convert_index_to_state(idx)
            assert convert_state_to_index(state1, state2) == idx
        print("Tests complete.")

    P, R = get_transition_and_reward_arrays(0.5)
    for discount in np.arange(.1, 1, .2):
        sdp = mdp.PolicyIteration(P, R, discount)
        sdp.setVerbose()
        start = clock()
        sdp.run()
        elapsed = clock() - start

    for discount in np.arange(.1, 1, .2):
        sdp = hmdp.mdp.QLearning(P, R, discount)
        sdp.setVerbose()
        start = clock()
        sdp.run()
        elapsed = clock() - start

    for discount in np.arange(.1, 1, .2):
        sdp = mdp.FiniteHorizon(P, R, discount, 50)
        sdp.setVerbose()
        start = clock()
        sdp.run()
        elapsed = clock() - start

    for discount in np.arange(.1, 1, .2):
        qlearner_stats = collections.defaultdict(list)
        ttt = hmdp.QLearning(P, R, discount)
        ttt.setVerbose()
        start = clock()
        ttt.run()
        elapsed = clock() - start
        for stats in ttt.run_stats:
            qlearner_stats['state'].append(stats['State'])
            qlearner_stats['action'].append(stats['Action'])
            qlearner_stats['reward'].append(stats['Reward'])
            qlearner_stats['error'].append(stats['Error'])
            qlearner_stats['time'].append(stats['Time'])
            qlearner_stats['alpha'].append(stats['Alpha'])
            qlearner_stats['epsilon'].append(stats['Epsilon'])
            qlearner_stats['max_v'].append(stats['Max V'])
            qlearner_stats['mean_v'].append(stats['Mean V'])
        qlearner_stats_df = pd.DataFrame(qlearner_stats)
        qlearner_stats_df.to_csv(f'{discount}_qlearner_stats_sdp')