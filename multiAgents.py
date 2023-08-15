# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util,time
from math import sqrt, log

from game import Agent



class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        # NOTE: this is an incomplete function, just showing how to get current state of the Env and Agent.

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
        Your minimax agent (question 1)
    """

    def getAction(self, gameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.
            Here are some method calls that might be useful when implementing minimax.
            gameState.getLegalActions(agentIndex):
                Returns a list of legal actions for an agent
                agentIndex=0 means Pacman, ghosts are >= 1
            gameState.generateSuccessor(agentIndex, action):
                Returns the successor game state after an agent takes an action
            gameState.getNumAgents():
                Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        search_depth=self.depth
        numOfAgents=gameState.getNumAgents()
                
        def maxValue(state,now_depth):
            outcome=[]

            if state.isWin() or state.isLose():
                outcome.append(self.evaluationFunction(state))
                outcome.append(["Stop"])
                return outcome
                    
            legalActions=state.getLegalActions(0)
            possibleValue={}
            for action in legalActions:
                successorGameState=state.generateSuccessor(0,action)
                possibleValue[action]=minValue(successorGameState,now_depth,1)
            bestScore = max(possibleValue.values())
            bestActions = [action for action in possibleValue.keys() if possibleValue[action] == bestScore]
            
            outcome.append(max(possibleValue.values()))
            outcome.append(bestActions)

            return outcome
                
        def minValue(state,now_depth,ghostID):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            if ghostID==numOfAgents-1:
                legalActions=state.getLegalActions(ghostID)
                possibleValue=[]
                for action in legalActions:
                    successorGameState=state.generateSuccessor(ghostID,action)
                    if now_depth==search_depth:
                        possibleValue.append(self.evaluationFunction(successorGameState))
                    else:
                        possibleValue.append(maxValue(successorGameState,now_depth+1)[0])
                return min(possibleValue)
            else:
                legalActions=state.getLegalActions(ghostID)
                valueOfAllGhostsMove=[]
                for action in legalActions:
                    successorGameState=state.generateSuccessor(ghostID,action)
                    valueOfAllGhostsMove.append(minValue(successorGameState,now_depth,ghostID+1))
                return min(valueOfAllGhostsMove)

        bestActions=maxValue(gameState,1)[1]
        chosenAction=random.choice(bestActions)
        return chosenAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
        Your minimax agent with alpha-beta pruning (question 2)
    """
    
    def getAction(self, gameState):
        """
            Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        search_depth=self.depth

        numOfAgents=gameState.getNumAgents()

        def maxValue(state,now_depth,alpha,beta):
            outcome=[]
            
            if state.isWin() or state.isLose():
                outcome.append(self.evaluationFunction(state))
                outcome.append(["Stop"])
                return outcome
            
            legalActions=state.getLegalActions(0)
            possibleValue={}
            for action in legalActions:
                successorGameState=state.generateSuccessor(0,action)
                tmp_value=minValue(successorGameState,now_depth,1,alpha,beta);
                alpha=max(alpha,tmp_value)
                if alpha>beta:
                    return (alpha,[action])

                else:
                    possibleValue[action]=tmp_value
                    
            bestScore = max(possibleValue.values())
            bestActions = [action for action in possibleValue.keys() if possibleValue[action] == bestScore]
                    
            outcome.append(bestScore)
            outcome.append(bestActions)

            return outcome

        def minValue(state,now_depth,ghostID,alpha,beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
                    
            if ghostID==numOfAgents-1:
                legalActions=state.getLegalActions(ghostID)
                possibleValue=[]

                for action in legalActions:
                    successorGameState=state.generateSuccessor(ghostID,action)
                    if now_depth==search_depth:
                        tmp_value=self.evaluationFunction(successorGameState)
                        beta=min(beta,tmp_value)
                        if beta<alpha:
                            return tmp_value
                        possibleValue.append(tmp_value)
                    else:
                        tmp_value=maxValue(successorGameState,now_depth+1,alpha,beta)[0]
                        beta=min(beta,tmp_value)
                        if beta<alpha:
                            return beta
                        else:
                            possibleValue.append(tmp_value)
                return min(possibleValue)
            else:
                legalActions=state.getLegalActions(ghostID)
                valueOfAllGhostsMove=[]
                for action in legalActions:
                    successorGameState=state.generateSuccessor(ghostID,action)
                    tmp_value=minValue(successorGameState,now_depth,ghostID+1,alpha,beta)
                    beta=min(tmp_value,beta)
                    if tmp_value<alpha:
                        return tmp_value
                            
                    valueOfAllGhostsMove.append(tmp_value)
                return min(valueOfAllGhostsMove)

        bestActions=maxValue(gameState,1,float("-inf"),float("inf"))[1]
        chosenAction=random.choice(bestActions)
        return chosenAction
        util.raiseNotDefined()

class MCTSAgent(MultiAgentSearchAgent):
    """
      Your MCTS agent with Monte Carlo Tree Search (question 3)
    """

    def getAction(self, gameState):
        class Node:
            def __init__(self,data):
                self.parent=None
                self.children={"East":None,\
                                "West":None,\
                                "North":None,\
                                "South":None,\
                                "Stop":None,\
                                "Left":None,\
                                "Reverse":None,\
                                "Right":None}
                self.state=data[0]
                self.assumedValue=float("inf")
                self.chosenNum=data[2]
                self.sum=data[1]
                self.denominater=data[2]

        def treePolicy(cgstree):
            reward=0
            state=cgstree.state
            while not state.isWin() and not state.isLose():
                action=SelectAction(state,cgstree)
                successorGameState=state.generateSuccessor(0,action)
                if successorGameState.getNumFood()<state.getNumFood():
                    reward=0.2
                else:
                    reward=-0.01
                #reward=reward+R
                if cgstree.children[action]==None:
                    cgstree.children[action]=Node([successorGameState,0,0])
                    return [cgstree.children[action],reward]
                cgstree=cgstree.children[action]
                state=successorGameState

            if state.isWin():
                return [cgstree,1]
            elif state.isLose():
                return [cgstree,0]
            
        def Simulation(cgs,reward0):
            reward=reward0
            state=cgs
            depth=1
            nowPacmanPosition=state.getPacmanPosition()

            while not state.isWin() and not state.isLose():
                legalActions=state.getLegalActions(0)
                legalActions=legalActions
                maxValue=float("-inf")
                chosenAction=None
                for action in legalActions:
                    successorGameState=state.generateSuccessor(0,action)
                    if successorGameState.isWin():
                        return 1-depth*0.0005
                    elif successorGameState.isLose():
                        return 0+depth*0.005
                    else:
                        foodDistance1,ghostDistance1=HeuristicFunction(state)
                        foodNum1=state.getNumFood()
                        foodDistance2,ghostDistance2=HeuristicFunction(successorGameState)
                        foodNum2=successorGameState.getNumFood()
                        newPacmanPosition=successorGameState.getPacmanPosition()
                        value=0

                        if ghostDistance1>2:
                            if foodDistance1<2:
                                value=0.1+(foodNum1-foodNum2)*0.05-depth*0.005
                            else:
                                value = 0.1+(foodDistance1-foodDistance2)*0.03-depth*0.005

                        else:
                            if foodDistance1<2:
                                value = 0.1+(foodNum1-foodNum2)*0.05 + \
                                    (ghostDistance1-ghostDistance2)*0.01-depth*0.005
                            else:
                                if foodNum1==1:
                                    value =0.1+(foodDistance1-foodDistance2)*0.035 + \
                                        (ghostDistance1-ghostDistance2)*0.02-depth*0.005###
                                else:
                                    value = 0.1+(foodDistance1-foodDistance2)*0.03 + \
                                        (ghostDistance1-ghostDistance2) *0.02-depth*0.005###
                        
                        if value>maxValue:
                            maxValue=value
                            chosenAction=action
                state=state.generateSuccessor(0,chosenAction)
                if reward+maxValue<0:
                    reward=0
                else:
                    reward=min(1,reward+maxValue)
                depth+=1
                if depth==6:
                    break    

            if state.isWin():
                return 1
            elif state.isLose():
                return 0
            return reward
        
        def BackPropagation(cgstree,reward):
            while cgstree!=None:
                cgstree.sum+=reward
                cgstree.chosenNum+=1
                cgstree.assumedValue=cgstree.sum/cgstree.chosenNum
                cgstree=cgstree.parent

        def SelectAction(cgs,cgstree):
            legalActions=cgs.getLegalActions(0)
            maxValue=float("-inf")
            for action in legalActions:
                if cgstree.children[action]==None:
                    return action
                if cgstree.children[action].chosenNum==0:
                    return action
                value=cgstree.children[action].assumedValue+sqrt(2*log(cgstree.chosenNum)/cgstree.children[action].chosenNum)
                if value>maxValue:
                    maxValue=value
                    chosenAction=action
            return chosenAction

        def HeuristicFunction(gamestate):
            pacmanPosition=gamestate.getPacmanPosition()
            currentfood=gamestate.getFood().asList()
            currentGhost=gamestate.getGhostPositions()
            minFoodDistance=float("inf")
            minGhostDistance=float("inf")
            for foodPosition in currentfood:
                if manhattanDistance(foodPosition,pacmanPosition)<minFoodDistance:
                    minFoodDistance=manhattanDistance(foodPosition,pacmanPosition)
            for ghostPosition in currentGhost:
                if manhattanDistance(ghostPosition,pacmanPosition)<minGhostDistance:
                    minGhostDistance=manhattanDistance(ghostPosition,pacmanPosition)
            return [minFoodDistance,minGhostDistance]

        nearestFoodDistance,nearestGhostDistance=HeuristicFunction(gameState)

        def astarDistance(gamestate,destination):
            start_time=time.time()
            state=gamestate
            s_ca={}
            Visited=[]
            frontier=util.PriorityQueue()
            frontier.push(gamestate,manhattanDistance(state.getPacmanPosition(),destination))
            s_ca[gamestate]=[manhattanDistance(state.getPacmanPosition(),destination),[]]
            while not frontier.isEmpty():
                
                nowState=frontier.pop()
                nowAction=s_ca[nowState][1]
                if time.time()-start_time>0.2:
                    return len(nowAction)+manhattanDistance(nowState.getPacmanPosition(),destination)
                Visited.append(nowState)
                if manhattanDistance(nowState.getPacmanPosition(),destination)==0:
                    return len(nowAction)
                legalActions=nowState.getLegalActions()
                for action in legalActions:
                    n_state=nowState.generateSuccessor(0,action)
                    n_cost=len(nowAction)+1+manhattanDistance(n_state.getPacmanPosition(),destination)
                    if n_state not in Visited:
                        frontier.push(n_state,n_cost)
                        if n_state in s_ca.keys():
                            if s_ca[n_state][0]>n_cost:
                                s_ca[n_state]=[n_cost,nowAction+[action]]
                        else:
                            s_ca[n_state]=[n_cost,nowAction+[action]]
                        
        
        if nearestFoodDistance<2 or nearestGhostDistance<4:
            cgstree=Node([gameState,0,1])
            start_time=time.time()
            while True:
                nowtree,reward0=treePolicy(cgstree)
                state=nowtree.state
                reward=Simulation(state,reward0)
                BackPropagation(nowtree,reward)
                if time.time()-start_time>0.2:
                    break
            maxValue=float("-inf")
            chosenAction=None
            for action in cgstree.children.keys():
                if cgstree.children[action]==None:
                    continue
                if cgstree.children[action].assumedValue>maxValue:
                    maxValue=cgstree.children[action].assumedValue
                    chosenAction=action
            #if chosenAction=="Stop":
            #    print(chosenAction)
            return chosenAction
        else:
            currentfood=gameState.getFood().asList()
            pacmanPosition=gameState.getPacmanPosition()
            
            for foodPosition in currentfood:
                if manhattanDistance(foodPosition,pacmanPosition)==nearestFoodDistance:
                    destination=foodPosition
                    break
            
            bestValue=float("inf")
            #print(destination)
            legalActions=gameState.getLegalActions()
            distance=astarDistance(gameState,destination)
            if distance>10:
                bestValue=float("inf")
                for action in legalActions:
                    successorGameState=gameState.generateSuccessor(0,action)
                    distance=astarDistance(successorGameState,destination)
                    if distance<bestValue:
                        chosenAction=action
                        bestValue=distance
            else:
                for action in legalActions:
                    successorGameState=gameState.generateSuccessor(0,action)
                    distance=manhattanDistance(successorGameState.getPacmanPosition(),destination)
                    if distance<bestValue:
                        chosenAction=action
                        bestValue=distance
                if(chosenAction=="Stop"):
                    bestValue=float("inf")
                    for action in legalActions:
                        successorGameState=gameState.generateSuccessor(0,action)
                        distance=astarDistance(successorGameState,destination)
                        if distance<bestValue:
                            chosenAction=action
                            bestValue=distance
            #print(chosenAction,' ',bestValue)
            return chosenAction
        util.raiseNotDefined()
