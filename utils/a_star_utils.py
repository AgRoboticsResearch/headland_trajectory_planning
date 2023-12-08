import numpy as np
import math
import heapq

BLOCK_COST = 100


def holonomic_motion_commands():

    # Action set for a King's move on the grid map
    holonomic_motion_command = [
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
    ]
    return holonomic_motion_command


def forward_holonomic_motion_commands():

    # Action set for a King's move on the grid map
    forward_holonomic_motion_command = [
        [-1, 0],  # left
        [0, 1],  # down
        [-1, 1],  # left down
        [1, 1],  # right down
        [1, 0],  # right
    ]
    return forward_holonomic_motion_command


# node for searching on the grid map
class HolonomicNode:
    def __init__(self, gridIndex, cost, parentIndex):
        self.gridIndex = gridIndex
        self.cost = cost
        self.parentIndex = parentIndex

    def get_index(self):
        return tuple([self.gridIndex[0], self.gridIndex[1]])


# check if the grid is valid
def holonomic_node_is_valid(neighbourNode: HolonomicNode, obstacles: np.ndarray):

    # Check if Node on obstacle
    weight, height = obstacles.shape
    # out of bound
    if (
        np.abs(neighbourNode.gridIndex[0]) >= weight
        or np.abs(neighbourNode.gridIndex[1]) >= height
    ):
        return False

    # on the obstacle
    if obstacles[neighbourNode.gridIndex[0]][neighbourNode.gridIndex[1]]:
        return False

    return True


# eucledian cost in grid search
def eucledian_cost(holonomicMotionCommand):
    # Compute Eucledian Distance between two nodes
    return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])
    # return 1


# holonomic cost on the grid map
def holonomic_costs_with_obstacles(
    goal_index: tuple, obstacles: np.ndarray, motion_type="King"
):
    """get cost matrices of the occupancy grids to the goal position
    Args:
        goal_index: index of the goal grid
        obstacles: boolean array of obstacles: True is filled with obstacle, False is empty
    Returns:
        cost_matrices: cost matrices of moving to the goal grid
    """
    gNode = HolonomicNode(goal_index, 0, goal_index)
    holonomicCost = np.copy(obstacles).astype("double")
    # king's motion
    if motion_type == "King":
        holonomicMotionCommand = holonomic_motion_commands()

    # soldier's motion only
    if motion_type == "Pawn":
        holonomicMotionCommand = forward_holonomic_motion_commands()
    openSet = {gNode.get_index(): gNode}
    closedSet = {}

    priorityQueue = []
    heapq.heappush(priorityQueue, (gNode.cost, gNode.get_index()))

    while True:
        if not openSet:
            break

        _, currentNodeIndex = heapq.heappop(priorityQueue)
        currentNode = openSet[currentNodeIndex]
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode

        for i in range(len(holonomicMotionCommand)):
            neighbourNode = HolonomicNode(
                [
                    currentNode.gridIndex[0] + holonomicMotionCommand[i][0],
                    currentNode.gridIndex[1] + holonomicMotionCommand[i][1],
                ],
                currentNode.cost + eucledian_cost(holonomicMotionCommand[i]),
                currentNodeIndex,
            )

            if not holonomic_node_is_valid(neighbourNode, obstacles):
                continue

            neighbourNodeIndex = neighbourNode.get_index()

            if neighbourNodeIndex not in closedSet:
                if neighbourNodeIndex in openSet:
                    if neighbourNode.cost < openSet[neighbourNodeIndex].cost:
                        openSet[neighbourNodeIndex].cost = neighbourNode.cost
                        openSet[
                            neighbourNodeIndex
                        ].parentIndex = neighbourNode.parentIndex
                        # heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
                else:
                    openSet[neighbourNodeIndex] = neighbourNode
                    heapq.heappush(
                        priorityQueue, (neighbourNode.cost, neighbourNodeIndex)
                    )

    holonomicCost[:, :] = np.inf
    for nodes in closedSet.values():
        holonomicCost[nodes.get_index()] = nodes.cost

    return holonomicCost
