from MultiAgentSystem import *
from Constants import *
import numpy


"""
    Represents a multi-agent system where the agents do not necessarily have all dependencies available at the beginning
    of the system
"""
class UnacceptableStartMultiAgentSystem(MultiAgentSystem):
    """
        Computes the baseline matrix for the provided matrix
        taking into account whether tasks' dependencies can also be executed
    """
    def _baseline(self, cm_matrix):
        baseline = numpy.zeros((1, cm_matrix.shape[CM_TASKS_AXIS]))

        for task_index in range(0, cm_matrix.shape[CM_TASKS_AXIS]):
            task_column = cm_matrix[:, task_index]
            for agent_index in range(0, len(task_column)):
                # Check whether the agent can execute the task
                if task_column[agent_index] == TRUE_VALUE:
                    # The agent can execute the task, now check whether the dependencies of the task can be executed
                    dependencies = self.dm[task_index]
                    dependencies_satisfied = True
                    for dependency_index in range(0, len(dependencies)):
                        if dependencies[dependency_index] == TRUE_VALUE \
                                and cm_matrix[agent_index, dependency_index] != TRUE_VALUE:
                            # dependency_index is a dependency and the agent cannot execute it
                            dependencies_satisfied = False

                    if dependencies_satisfied:
                        # Add one more agent to the count of agents able to execute the task
                        baseline[0, task_index] = baseline[0, task_index] + 1

        return baseline


"""
    Represents a multi-agent system where leaf nodes cannot be transferred between agents
"""
class NoLeafNodeTransferMultiAgentSystem(UnacceptableStartMultiAgentSystem):
    @classmethod
    def _create_tm(cls, dm, ecm):
        # Create the TM normally as in the parent class
        tm = super()._create_tm(dm, ecm)

        # Now unset the TM where tasks are leaf nodes
        dependencies_of = dm.sum(1)
        for task_index in range(0, ecm.shape[CM_TASKS_AXIS]):
            if dependencies_of[task_index] == 0 and ecm.sum(0)[task_index] > 0:
                for agent_index in range(0, ecm.shape[CM_AGENTS_AXIS]):
                    tm[agent_index, task_index] = 0

        return tm


"""
    Represents a multi-agent system where the ECM is set in accordance with the 
    transitive closure of the DM
"""
class AcceptableStartMultiAgentSystem(MultiAgentSystem):
    """
        Creates a random ECM for the MAS with the conditions of an 'Acceptable Start'
    """
    @classmethod
    def _create_random_ecm(cls, dm, num_agents, num_tasks):
        ecm = super()._create_random_ecm(dm, num_agents, num_tasks)

        for agent_index in range(0, ecm.shape[CM_AGENTS_AXIS]):
            for task_index in range(0, ecm.shape[CM_TASKS_AXIS]):
                # Check which tasks the agent can perform
                if ecm[agent_index, task_index] == TRUE_VALUE:
                    # This task exists in the agent, set its dependencies via the DM
                    dependencies_of_t = dm[task_index]
                    for dependency_index in range(0, len(dependencies_of_t)):
                        if dependencies_of_t[dependency_index] == TRUE_VALUE:
                            # The task at index dependency_index is a dependency of the task at task_index
                            # Set the agent to also be able to do dependency_index
                            ecm[agent_index, dependency_index] = TRUE_VALUE

        return ecm

    """
        Returns True if the PCM cell should be set, False otherwise
        Computes this based on whether the task exists locally at the agent already in the ECM, and whether it,
        and all its dependencies, can be transferred in from another agent
    """
    def _pcm_cell(self, agent_index, task_index):
        if self.ecm[agent_index, task_index] == TRUE_VALUE:
            # This task is available locally
            return True
        elif self._can_transfer(task_index=task_index):
            # The task is not available locally, but can be transferred in from another agent in the system
            can_get_dependencies = True
            for i in range(0, self.ecm.shape[CM_TASKS_AXIS]):
                if self.dm[task_index, i] == TRUE_VALUE:
                    # i is a dependency of the requested task
                    can_get_dependencies = can_get_dependencies and \
                                           (self.ecm[agent_index, i] == TRUE_VALUE or self._can_transfer(task_index=i))
            return can_get_dependencies
        else:
            return False


"""
    Represents a multi-agent system without Intelligence Transfer
    PCM := ECM, TM := zero matrix
"""
class NoIntelligenceTransferMultiAgentSystem(MultiAgentSystem):
    def _compute_pcm(self):
        self.pcm = numpy.copy(self.ecm)

    @classmethod
    def _create_tm(cls, dm, ecm):
        return numpy.zeros(ecm.shape)


"""
    Represents a MAS where the randomisation is weighted in favour of some weighted value
"""
class WeightedMultiAgentSystem(MultiAgentSystem):
    @classmethod
    def _binary_2d_randomize(cls, matrix):
        for i in range(0, matrix.shape[0]):
            for j in range(0, matrix.shape[1]):
                matrix[i][j] = random.choices([0, 1], weights=[1, 1], k=1)[0]

        return matrix


"""
    Represents a non-randomised MAS with the specific component matrices as defined
"""
class SpecificMultiAgentSystem(UnacceptableStartMultiAgentSystem):
    @classmethod
    def _create_random_ecm(cls, dm, num_agents, num_tasks):
        return numpy.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ])

    @classmethod
    def _create_random_dm(cls, num_tasks):
        return numpy.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    @classmethod
    def _create_tm(cls, dm, ecm):
        return ecm
