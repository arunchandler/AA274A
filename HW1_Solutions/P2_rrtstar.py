import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, line_line_intersection

class RRT(object):
    """ Represents a motion planning problem to be solved using the RRT algorithm"""
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, obstacles):
        self.statespace_lo = np.array(statespace_lo)    # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = np.array(statespace_hi)    # state space upper bound (e.g., [5, 5])
        self.x_init = np.array(x_init)                  # initial state
        self.x_goal = np.array(x_goal)                  # goal state
        self.obstacles = obstacles                      # obstacle set (line segments)
        self.path = None        # the final path as a list of states

    def is_free_motion(self, obstacles, x1, x2):
        """
        Subject to the robot dynamics, returns whether a point robot moving
        along the shortest path from x1 to x2 would collide with any obstacles
        (implemented as a "black box")

        Inputs:
            obstacles: list/np.array of line segments ("walls")
            x1: start state of motion
            x2: end state of motion
        Output:
            Boolean True/False
        """
        raise NotImplementedError("is_free_motion must be overriden by a subclass of RRT")

    def find_nearest(self, V, x):
        """
        Given a list of states V and a query state x, returns the index (row)
        of V such that the steering distance (subject to robot dynamics) from
        V[i] to x is minimized

        Inputs:
            V: list/np.array of states ("samples")
            x - query state
        Output:
            Integer index of nearest point in V to x
        """
        raise NotImplementedError("find_nearest must be overriden by a subclass of RRT")

    def steer_towards(self, x1, x2, eps):
        """
        Steers from x1 towards x2 along the shortest path (subject to robot
        dynamics). Returns x2 if the length of this shortest path is less than
        eps, otherwise returns the point at distance eps along the path from
        x1 to x2.

        Inputs:
            x1: start state
            x2: target state
            eps: maximum steering distance
        Output:
            State (numpy vector) resulting from bounded steering
        """
        raise NotImplementedError("steer_towards must be overriden by a subclass of RRT")

    def solve(self, eps, max_iters=1000, goal_bias=0.05, shortcut=False, gamma=5.0):
        state_dim = len(self.x_init)

        # Initialize tree
        V = np.zeros((max_iters + 1, state_dim))
        V[0, :] = self.x_init
        P = -np.ones(max_iters + 1, dtype=int)
        costs = np.zeros(max_iters + 1)  # track cost to reach each node
        n = 1
        success = False

        for i in range(max_iters):
            if np.random.rand() < goal_bias:
                x_rand = self.x_goal
            else:
                x_rand = self.statespace_lo + np.random.rand(state_dim) * (self.statespace_hi - self.statespace_lo)
            nn_idx = self.find_nearest(V[:n, :], x_rand)
            x_near = V[nn_idx, :]
            x_new = self.steer_towards(x_near, x_rand, eps)
            if self.is_free_motion(self.obstacles, x_near, x_new):
                r_n = gamma * (np.log(n) / n)**(1 / state_dim)
                neighbor_idxs = [j for j in range(n) if np.linalg.norm(V[j, :] - x_new) < r_n]

                # Select the best parent within r_n
                min_cost = costs[nn_idx] + np.linalg.norm(x_near - x_new)
                best_parent = nn_idx
                for j in neighbor_idxs:
                    potential_cost = costs[j] + np.linalg.norm(V[j, :] - x_new)
                    if self.is_free_motion(self.obstacles, V[j, :], x_new) and potential_cost < min_cost:
                        min_cost = potential_cost
                        best_parent = j

                # Add x_new to the tree
                V[n, :] = x_new
                P[n] = best_parent
                costs[n] = min_cost
                n += 1

                # Rewire the tree
                for j in neighbor_idxs:
                    if j != best_parent and costs[n - 1] + np.linalg.norm(V[j, :] - x_new) < costs[j]:
                        if self.is_free_motion(self.obstacles, x_new, V[j, :]):
                            P[j] = n - 1  # Update parent
                            costs[j] = costs[n - 1] + np.linalg.norm(V[j, :] - x_new)

                # Check if we reached the goal
                if (x_new == self.x_goal).all():
                    success = True
                    break

        if success:
            solution_idxs = [n - 1]
            while P[solution_idxs[0]] != -1:
                solution_idxs.insert(0, P[solution_idxs[0]])
            self.path = V[solution_idxs, :]

        # Plotting as before
        plt.figure()
        self.plot_problem()
        self.plot_tree(V, P, color="blue", linewidth=.5, label="RRT* tree", alpha=0.5)
        if success:
            if shortcut:
                self.plot_path(color="purple", linewidth=2, label="Original solution path")
                self.shortcut_path()
                self.plot_path(color="green", linewidth=2, label="Shortcut solution path")
            else:
                self.plot_path(color="green", linewidth=2, label="Solution path")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
            plt.scatter(V[:n, 0], V[:n, 1])
        else:
            print("Solution not found!")

        return success

    def plot_problem(self):
        plot_line_segments(self.obstacles, color="red", linewidth=2, label="obstacles")
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
        plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
        plt.axis('scaled')

    def shortcut_path(self):
        """
        Iteratively removes nodes from solution path to find a shorter path
        which is still collision-free.
        Input:
            None
        Output:
            None, but should modify self.path
        """
        ########## Code starts here ##########
        shortcut = True
        while shortcut:
            shortcut = False
            for i in range(1,len(self.path)-1):
                if self.is_free_motion(self.obstacles, self.path[i-1], self.path[i+1]):
                    shortcut = True
                    self.path = np.vstack([self.path[:i], self.path[i+1:]])
                    break

        ########## Code ends here ##########

class GeometricRRT(RRT):
    """
    Represents a geometric planning problem, where the steering solution
    between two points is a straight line (Euclidean metric)
    """

    def find_nearest(self, V, x):
        # Consult function specification in parent (RRT) class.
        ########## Code starts here ##########
        # Hint: This should take 1-3 line.
        return np.argmin(np.linalg.norm(x - V, axis=1))
        ########## Code ends here ##########
        pass

    def steer_towards(self, x1, x2, eps):
        # Consult function specification in parent (RRT) class.
        ########## Code starts here ##########
        # Hint: This should take 1-4 line.
        return x1 + (x2 - x1)*min(eps/np.linalg.norm(x2 - x1), 1)
        ########## Code ends here ##########
        pass

    def is_free_motion(self, obstacles, x1, x2):
        motion = np.array([x1, x2])
        for line in obstacles:
            if line_line_intersection(motion, line):
                return False
        return True

    def plot_tree(self, V, P, **kwargs):
        plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

    def plot_path(self, **kwargs):
        path = np.array(self.path)
        plt.plot(path[:,0], path[:,1], **kwargs)
