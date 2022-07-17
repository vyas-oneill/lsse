from Constants import *
import numpy
import random


"""
    Represents a multi-agent system implementing the ITM
"""
class MultiAgentSystem(object):
    """
        Creates a new multi-agent system with the provided ECM, transitive closure of DM and TM
    """
    def __init__(self, ecm, dm, tm):
        self.ecm = ecm
        self.dm = dm
        self.tm = tm
        self._compute_pcm()

    """
        Implements the Can-Transfer function
        Returns True if the provided task can be transferred by an agent in the system, False otherwise
    """
    def _can_transfer(self, task_index):
        result = False
        agents_capable_of_transferring_task = self.tm[:, task_index]
        if TRUE_VALUE in agents_capable_of_transferring_task:
            result = True

        return result

    """
        Implements the PCM-Cell function
        Returns True if the PCM cell should be set, False otherwise
        Computes this based on whether the task exists locally at the agent already in the ECM, and whether it can
        be transferred in from another agent. DEPENDENCIES ARE NOT CONSIDERED
    """
    def _pcm_cell(self, agent_index, task_index):
        if self.ecm[agent_index, task_index] == TRUE_VALUE:
            # This task is available locally
            return True
        elif self._can_transfer(task_index=task_index):
            # The task is not available locally, but can be transferred in from another agent in the system
            return True
        else:
            return False

    """
        Computes the PCM of the MAS
    """
    def _compute_pcm(self):
        self.pcm = numpy.copy(self.ecm)

        for agent_index in range(0, self.pcm.shape[CM_AGENTS_AXIS]):
            for task_index in range(0, self.pcm.shape[CM_TASKS_AXIS]):
                if self._pcm_cell(agent_index=agent_index, task_index=task_index):
                    self.pcm[agent_index, task_index] = TRUE_VALUE

    """
        Returns a baseline matrix based on the provided capability matrix
    """
    def _baseline(self, cm_matrix):
        baseline = numpy.zeros((1, cm_matrix.shape[CM_TASKS_AXIS]))

        for i in range(0, cm_matrix.shape[CM_TASKS_AXIS]):
            task_column = cm_matrix[:, i]
            for agent_index in task_column:
                if agent_index == TRUE_VALUE:
                    # Add to the counter of agents that can execute this task
                    baseline[0, i] = baseline[0, i] + 1

        return baseline

    """
        Computes the ECM baseline (B Matrix)
    """
    def ecm_baseline(self):
        return self._baseline(self.ecm)

    """
        Computes the PCM baseline (B-Star Matrix)
    """
    def pcm_baseline(self):
        return self._baseline(self.pcm)

    def __str__(self):
        s = ("*" * 50) + \
            f"\n> {type(self).__name__}\nECM\n{self.ecm}\nDM\n{self.dm}\nTM\n{self.tm}\nPCM\n{self.pcm}\n" + \
            f"ECM Baseline\n{self.ecm_baseline()}\nPCM Baseline\n{self.pcm_baseline()}\n"
        if numpy.array_equal(self.ecm, self.pcm):
            s = s + "ECM = PCM\n"
        s = s + ("*" * 50)
        return s

    def __eq__(self, other):
        return numpy.array_equal(self.ecm, other.ecm) and \
               numpy.array_equal(self.dm, other.dm) and \
               numpy.array_equal(self.tm, other.tm)

    """
        Implements the transitive closure using the Floyd-Warshall algorithm
        Ref: https://stackoverflow.com/questions/13716540/transitive-relation-in-an-adjacency-matrix
    """
    @classmethod
    def _transitive_closure(cls, matrix):
        tc = numpy.copy(matrix)
        for k in range(0, tc.shape[0]):
            for i in range(0, tc.shape[0]):
                for j in range(0, tc.shape[0]):
                    if tc[i][k] and tc[k][j]:
                        tc[i][j] = TRUE_VALUE

        return tc

    """
        Pseudorandomly populates the provided 2D matrix with binary values
    """
    @classmethod
    def _binary_2d_randomize(cls, matrix):
        temp = numpy.copy(matrix)
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                temp[i][j] = random.randint(0, 1)

        return temp

    """
        Determines whether the graph represented by the dependency matrix has loops
    """
    @classmethod
    def _dm_has_loops(cls, dm):
        for x in range(0, dm.shape[0]):
            if dm[x, x] == TRUE_VALUE:
                # Cycle detected, reject
                return True

        return False

    """
        Creates a random ECM guaranteed not to be a zero matrix
    """
    @classmethod
    def _create_random_ecm(cls, dm, num_agents, num_tasks):
        ecm = numpy.zeros([num_agents, num_tasks])
        while numpy.array_equal(ecm, numpy.zeros([num_agents, num_tasks])):
            ecm = cls._binary_2d_randomize(ecm)

        return ecm

    """
        Creates a random DM guaranteed to be a transitive closure and not to have any loops
    """
    @classmethod
    def _create_random_dm(cls, num_tasks):
        # Generate a random DM that has no loops
        dm = None
        has_loops = True
        while has_loops:
            dm = cls._binary_2d_randomize(numpy.zeros([num_tasks, num_tasks]))
            dm = cls._transitive_closure(dm)
            has_loops = cls._dm_has_loops(dm)

        return dm

    """
        Creates a naive TM equivalent to the ECM
    """
    @classmethod
    def _create_tm(cls, dm, ecm):
        return numpy.copy(ecm)

    """
        Generates a randomised instance of the MAS given its inherent characteristics 
        which may be defined in child classes
    """
    @classmethod
    def create_random_instance(cls, num_agents, num_tasks):
        # Generate a random ECM
        dm = cls._create_random_dm(num_tasks)
        ecm = cls._create_random_ecm(dm, num_agents, num_tasks)
        tm = cls._create_tm(dm, ecm)

        return cls(ecm, dm, tm)
