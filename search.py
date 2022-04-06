# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import searchAgents


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    """print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))"""


    # initialize path, toExpand, expanded node lists
    """startState = problem.getStartState()
    firstLevel = problem.getSuccessors(startState)
    path = util.Stack()
    path.push([startState, None, None])
    toExpand = util.Stack()
    expanded = util.Stack()
    expanded.push(startState)
    print(firstLevel)
    for i in firstLevel:
        toExpand.push(i)

    reachedGoal = False

    while not reachedGoal:
        successors = problem.getSuccessors(path.list[-1][0])
        nodeToExpand = toExpand.list[-1]
        if nodeToExpand not in successors:
            path.pop()
        else:
            if nodeToExpand[0] in expanded.list:
                toExpand.pop()
            else:
                expanded.push(nodeToExpand[0])
                if problem.isGoalState(nodeToExpand[0]):
                    path.push(nodeToExpand)
                    reachedGoal = True
                else:
                    successors = problem.getSuccessors(nodeToExpand[0])
                    expandable = []
                    for i in range(len(successors)):
                        if successors[i][0] not in expanded.list:
                            expandable.append(successors[i])
                    if expandable:
                        path.push(nodeToExpand)
                        toExpand.pop()
                        for i in range(len(expandable)):
                            toExpand.push(expandable[i])
                    else:
                        toExpand.pop()
    result = [path.list[i + 1][1] for i in range(len(path.list) - 1)]
    return result
    # util.raiseNotDefined()"""
    start_state = problem.getStartState()
    stack = util.Stack()
    for successor in problem.getSuccessors(start_state):
        action = successor[1]
        state = successor[0]
        stack.push((state, [action]))
    expanded = [start_state]
    while not stack.isEmpty():
        current_state, actions = stack.pop()
        expanded.append(current_state)
        if problem.isGoalState(current_state):
            return actions
        for successor in problem.getSuccessors(current_state):
            action = successor[1]
            state = successor[0]
            if state not in expanded:
                new_actions = actions.copy()
                new_actions.append(action)
                stack.push((state, new_actions))
                #expanded.append(state)



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Initialize path, queue and visited
    file = util.Queue()
    startState = problem.getStartState()
    visited = [startState]
    successors = problem.getSuccessors(startState)
    for successor in successors:
        succ_state, path = successor[0], [successor[1]]
        file.push((path, succ_state))
        visited.append(succ_state)
        # visited.append(successor)

    while not file.isEmpty():
        path, current_state = file.pop()
        if not problem.isGoalState(current_state):
            for successor in problem.getSuccessors(current_state):
                succ_state = successor[0]
                direction = successor[1]
                if succ_state not in visited:
                    path_copy = path.copy()
                    path_copy.append(direction)
                    file.push((path_copy, succ_state))
                    visited.append(succ_state)
                    # visited.append(successor)

        elif current_state:
            print(path)
            return path


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # Define priority queue
    priorityQ = util.PriorityQueue()
    start_state = problem.getStartState()
    visited = []
    priorityQ.update((start_state, []), 0)
    # the action_list is a dictionnary that maps a state and the current lowest cost of actions to it
    """action_list = {start_state: []}
    actions = action_list[start_state].copy()
    for successor in problem.getSuccessors(start_state):
        item = successor[0]
        actions.append(successor[1])
        priority = problem.getCostOfActions(actions)
        priorityQ.update(item, priority)
        action_list[item] = actions.copy()
        actions.pop()
    visited = [start_state]"""

    while not priorityQ.isEmpty():
        current_node,  actions = priorityQ.pop()
        #actions = action_list[current_node].copy()
        if current_node in visited:
            continue
        if problem.isGoalState(current_node):
            return actions
        visited.append(current_node)
        for successor in problem.getSuccessors(current_node):
            # current priority of successor
            child_state = successor[0]
            action = successor[1]
            if child_state in visited:
                continue
            path = actions.copy()
            path.append(action)
            cost = problem.getCostOfActions(path)
            priorityQ.update((child_state, path), cost)
            # if action legal
            """if current_priority != 999999 and successor[0] not in action_list.keys():  # != start_state
                # if successor has a path already
                action_list[successor[0]] = actions.copy()
                if problem.isGoalState(successor[0]):
                    return action_list[successor[0]]
                priorityQ.update(successor[0], current_priority)
            actions.pop()"""

    #return action_list[problem.goal]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # shortest_path_cost = lambda x: problem.getCostOfActions(globals()[uniformCostSearch(searchAgents.PositionSearchProblem())])
    # print(problem.costFn((5, 2, ())))
    if type(problem) is searchAgents.PositionSearchProblem:
        costFn = problem.costFn
        problem.costFn = lambda x: heuristic(x, problem) + costFn(x)
        return uniformCostSearch(problem)

    else: #if type(problem) is searchAgents.CornersProblem:

        hn = lambda x: heuristic(x, problem)
        priorityQ = util.PriorityQueue()
        goal_state = None
        start_state = problem.getStartState()
        # the action_list a dictionnary that maps a state and the current lowest cost of actions to it
        action_list = {start_state: {"cost": 0, "sequence": []}}
        path = action_list[start_state]

        # initialize queue
        for successor in problem.getSuccessors(start_state):
            succ_state = successor[0]
            parent_cost = path["cost"]
            child_cost = parent_cost + 1 + hn(succ_state)
            sequence = path["sequence"].copy()
            sequence.append(successor[1])
            # priority = problem.getCostOfActions(actions) + hn(item)
            priorityQ.update(succ_state, child_cost)
            # print(sequence, cost)
            # print(action_list[(5,1)])
            # print("Item: {}".format(item))
            action_list[succ_state] = {"cost": child_cost, "sequence": sequence.copy()}
            sequence.pop()

        while not priorityQ.isEmpty():
            current_node = priorityQ.pop()
            if problem.isGoalState(current_node):
                print("goal reached: {}".format(current_node))
                print(action_list[current_node]["sequence"])
                return action_list[current_node]["sequence"]
            path = action_list[current_node]
            for successor in problem.getSuccessors(current_node):
                succ_state = successor[0]
                # current priority of successor
                if succ_state not in action_list.keys():
                    parent_cost = path["cost"]
                    sequence = path["sequence"].copy()
                    sequence.append(successor[1])
                    child_cost = hn(succ_state) + parent_cost + 1
                    priorityQ.update(succ_state, child_cost)
                    action_list[succ_state] = {"cost": child_cost, "sequence": sequence.copy()}
                    sequence.pop()

        # return action_list[problem.goal]
    # util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
