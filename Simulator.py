import math
import sys
import threading
from threading import Thread
import time

import matplotlib.pyplot as plt
import statistics
import numpy

from MultiAgentSystem import *
from ChildMultiAgentSystems import *
from Constants import *

"""
    Aggregates the results for a MAS experiment
"""
class Aggregator(object):
    def __init__(self, num_agents, num_tasks, max_size=-1, reliability_steps=100, evolution_time=5.0):
        self.num_agents = num_agents
        self.num_tasks = num_tasks
        self.max_size = max_size
        self.reliability_steps = reliability_steps
        self.evolution_time = evolution_time

        self.multi_agent_systems = []
        self.ecm_baseline_states = []
        self.pcm_baseline_states = []
        self.num_new_states = []
        self.state_improvement_factors = []
        self.ecm_reliability = []
        self.pcm_reliability = []
        self.reliability_improvement_factors = []
        self.ecm_task_based_reliabilities = []
        self.pcm_task_based_reliabilities = []
        self.ecm_mean_task_based_reliabilities = []
        self.pcm_mean_task_based_reliabilities = []
        self.failure_to_acceptable = []

        for task_index in range(0, num_tasks):
            self.ecm_task_based_reliabilities.append([])
            self.pcm_task_based_reliabilities.append([])

        self.finalized = False
        self.lock = threading.Lock()

    """
        Computes the reliability function R(t) of a task
    """
    def _R_T(self, t, agent_redundancy):
        return 1 - math.pow((1 - math.pow(math.e, -t)), agent_redundancy)

    """
        Computes the system reliability function defined as the product of reliability functions
        of all tasks in the MAS
    """
    def _system_reliability_function(self, baseline_matrix):
        return self._task_based_reliability_function(baseline_matrix).prod(0)

    """
        Computes the task-based reliability function for each task in the MAS
    """
    def _task_based_reliability_function(self, baseline_matrix):
        task_reliabilities = numpy.zeros([baseline_matrix.shape[CM_TASKS_AXIS], self.reliability_steps])
        for task_index in range(0, baseline_matrix.shape[CM_TASKS_AXIS]):
            for step in range(0, self.reliability_steps):
                # Compute the value of the reliability function for the current task at each time interval
                task_reliabilities[task_index, step] = self._R_T(step * self.evolution_time / self.reliability_steps,
                                                                 baseline_matrix[0, task_index])
                if baseline_matrix[0, task_index] == 0:
                    pass
        return task_reliabilities

    """
        Returns a tuple (value, count_with_value) from a list of values
    """
    def _create_histogram(self, list_of_values):
        unique, counts = numpy.unique(numpy.array(list_of_values), return_counts=True)
        result = dict(zip(unique, counts))
        return list(result.keys()), list(result.values())

    """
        Adds the data from the provided MAS to the aggregate reliability metrics
    """
    def _aggregate_reliability(self, mas):
        ecm_baseline_matrix = mas.ecm_baseline()
        pcm_baseline_matrix = mas.pcm_baseline()

        ecm_reliability = self._system_reliability_function(ecm_baseline_matrix)
        pcm_reliability = self._system_reliability_function(pcm_baseline_matrix)

        self.ecm_reliability.append(ecm_reliability)
        self.pcm_reliability.append(pcm_reliability)

        ecm_tbrfs = self._task_based_reliability_function(ecm_baseline_matrix)
        pcm_tbrfs = self._task_based_reliability_function(pcm_baseline_matrix)
        for task_index in range(0, self.num_tasks):
            self.ecm_task_based_reliabilities[task_index].append(ecm_tbrfs[task_index])
            self.pcm_task_based_reliabilities[task_index].append(pcm_tbrfs[task_index])

        self.ecm_mean_task_based_reliabilities.append(ecm_tbrfs.mean(axis=0))
        self.pcm_mean_task_based_reliabilities.append(pcm_tbrfs.mean(axis=0))

    """
        Adds the data from the provided MAS to the aggregate fault tolerance metrics
    """
    def _aggregate_ft(self, mas):
        # Compute ECM and PCM states
        ecm_baseline_matrix = mas.ecm_baseline()
        pcm_baseline_matrix = mas.pcm_baseline()
        num_ecm_states = 0
        num_pcm_states = 0
        num_failure_to_acceptable = 0
        for task_index in range(0, ecm_baseline_matrix.shape[CM_TASKS_AXIS]):
            num_ecm_states += ecm_baseline_matrix[0, task_index]
            num_pcm_states += pcm_baseline_matrix[0, task_index]
            if ecm_baseline_matrix[0, task_index] == 0 and pcm_baseline_matrix[0, task_index] > 0:
                num_failure_to_acceptable += 1

        self.ecm_baseline_states.append(num_ecm_states)
        self.pcm_baseline_states.append(num_pcm_states)
        self.num_new_states.append(num_pcm_states - num_ecm_states)
        if num_ecm_states > 0:
            # The original MAS in its ECM configuration was capable of at least one task
            self.state_improvement_factors.append(num_pcm_states/num_ecm_states)
        else:
            # The original MAS was in a failure state
            # Do not record a change in improvement factor for this MAS
            pass

        self.failure_to_acceptable.append(num_failure_to_acceptable)

    """
        Returns True if the aggregator could accept the MAS
        False if the aggregator was finalized
    """
    def accept(self, mas, accept_duplicates=True):
        with self.lock:
            if not self.finalized:
                if accept_duplicates or mas not in self.multi_agent_systems:
                    if self.max_size == -1 or len(self) < self.max_size:
                        self._aggregate_ft(mas)
                        self._aggregate_reliability(mas)
                        self.multi_agent_systems.append(mas)
                        return True

            return False

    """
        Accepts a MAS into the aggregate results on the condition that it is unique
        Returns True if the aggregator could accept the MAS
        False if the aggregator was finalized or the MAS was a duplicate
    """
    def accept_unique(self, mas):
        return self.accept(mas, accept_duplicates=False)

    """
        Finalises the results of the experiment in preparation for output
    """
    def finalize(self):
        with self.lock:
            self.finalized = True

    """
        Returns the number of MASs in the aggregator
    """
    def count(self):
        return len(self.multi_agent_systems)

    def __len__(self):
        return self.count()

    """
        Returns the mean ECM task redundancy        
    """
    def av_ecm_task_redundancy(self):
        return statistics.mean([x/self.num_tasks for x in self.ecm_baseline_states])

    """
        Returns the median ECM task redundancy
    """
    def median_ecm_task_redundancy(self):
        return statistics.median([x / self.num_tasks for x in self.ecm_baseline_states])

    """
        Returns the standard deviation of the ECM task redundancy
    """
    def std_dev_ecm_task_redundancy(self):
        return statistics.stdev([x / self.num_tasks for x in self.ecm_baseline_states])

    """
        Returns the mean PCM task redundancy
    """
    def av_pcm_task_redundancy(self):
        return statistics.mean([x/self.num_tasks for x in self.pcm_baseline_states])

    """
        Returns the median PCM task redundancy
    """
    def median_pcm_task_redundancy(self):
        return statistics.median([x / self.num_tasks for x in self.pcm_baseline_states])

    """
        Returns the standard deviation of the PCM task redundancy
    """
    def std_dev_pcm_task_redundancy(self):
        return statistics.stdev([x / self.num_tasks for x in self.pcm_baseline_states])

    """
        Returns the mean number of new states
    """
    def av_num_new_states(self):
        return statistics.mean(self.num_new_states)

    """
        Returns the median number of new states
    """
    def median_num_new_states(self):
        return statistics.median(self.num_new_states)

    """
        Returns the standard deviation of the number of new states
    """
    def std_dev_num_new_states(self):
        return statistics.stdev(self.num_new_states)

    """
        Returns the mean state improvement factor
    """
    def av_state_improvement_factor(self):
        return statistics.mean(self.state_improvement_factors) if len(self.state_improvement_factors) > 0 else -1

    """
        Returns the median state improvement factor
    """
    def median_state_improvement_factor(self):
        return statistics.median(self.state_improvement_factors) if len(self.state_improvement_factors) > 0 else -1

    """
        Returns the standard deviation of the state improvement factor
    """
    def std_dev_state_improvement_factor(self):
        return statistics.stdev(self.state_improvement_factors) if len(self.state_improvement_factors) > 0 else -1

    """
        Returns the mean number of states which transitioned from failure to acceptable states
    """
    def av_failure_to_acceptable(self):
        return statistics.mean(self.failure_to_acceptable)

    """
        Returns the median number of states which transitioned from failure to acceptable states
    """
    def median_failure_to_acceptable(self):
        return statistics.median(self.failure_to_acceptable)

    """
        Returns the standard deviation of the number of states which transitioned from failure to acceptable states
    """
    def std_dev_failure_to_acceptable(self):
        return statistics.stdev(self.failure_to_acceptable)

    """
        Returns the range of the mean ECM reliability function
    """
    def ecm_reliability_range(self):
        return numpy.array(self.ecm_reliability).mean(0).min(0), numpy.array(self.ecm_reliability).mean(0).max(0)

    """
        Returns the range of the mean PCM reliability function
    """
    def pcm_reliability_range(self):
        return numpy.array(self.pcm_reliability).mean(0).min(0), numpy.array(self.pcm_reliability).mean(0).max(0)

    """
        Returns the ECM task-based reliability for the given MAS
    """
    def ecm_task_based_reliability(self, mas_index):
        return self._task_based_reliability_function(self.multi_agent_systems[mas_index].ecm_baseline())

    """
        Returns the PCM task-based reliability for the given MAS
    """
    def pcm_task_based_reliability(self, mas_index):
        return self._task_based_reliability_function(self.multi_agent_systems[mas_index].pcm_baseline())

    def __str__(self):
            return f"Aggregated Results for {len(self)} MASs with {self.num_agents} agents and {self.num_tasks} tasks\n" + \
                   ('*' * 50) + \
                    f"\nMean ECM task redundancy: {self.av_ecm_task_redundancy()}" + \
                    f"\nMedian ECM task redundancy: {self.median_ecm_task_redundancy()}" + \
                    f"\nStdDev ECM task redundancy: {self.std_dev_ecm_task_redundancy() if len(self) > 1 else 'N/a'}\n" + \
                    f"\nMean PCM task redundancy: {self.av_pcm_task_redundancy()}" + \
                    f"\nMedian PCM task redundancy: {self.median_pcm_task_redundancy()}" + \
                    f"\nStdDev PCM task redundancy: {self.std_dev_pcm_task_redundancy() if len(self) > 1 else 'N/a'}\n" + \
                    f"\nMean new states per MAS: {self.av_num_new_states()}" + \
                    f"\nMedian new states per MAS: {self.median_num_new_states()}" + \
                    f"\nStdDev new states per MAS: {self.std_dev_num_new_states() if len(self) > 1 else 'N/a'}\n" + \
                    f"\nMean state improvement factor : {self.av_state_improvement_factor()}" + \
                    f"\nMedian state improvement factor : {self.median_state_improvement_factor()}" + \
                    f"\nStdDev state improvement factor : {self.std_dev_state_improvement_factor() if len(self) > 1 else 'N/a'}\n" + \
                    f"\nMean failure to acceptable: {self.av_failure_to_acceptable()}" + \
                    f"\nMedian failure to acceptable: {self.median_failure_to_acceptable()}" + \
                    f"\nStdDev failure to acceptable: {self.std_dev_failure_to_acceptable() if len(self) > 1 else 'N/a'}\n" + \
                    f"\nRange of ECM Reliability: [{self.ecm_reliability_range()}]" + \
                    f"\nRange of PCM Reliability: [{self.pcm_reliability_range()}]" + \
                   "\n" + ('*' * 50)

"""
    Contains the logic for plotting graphical output from MAS experiments
"""
class Plotter:
    """
        Plots the graphical display from the given aggregator's experimental results
    """
    @classmethod
    def plot(cls, aggregator):
        if not aggregator.finalized:
            aggregator.finalize()

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.suptitle(f"ITM MAS Simulation ({aggregator.num_agents} agents, {aggregator.num_tasks} tasks)")
        fig.tight_layout(pad=3.0)

        # State Improvement Factor per MAS
        axs[0, 0].scatter(*aggregator._create_histogram(aggregator.num_new_states), color="black")
        axs[0, 0].set_xlabel("Number of New Acceptable States")
        axs[0, 0].set_ylabel("Count of MAS")
        axs[0, 0].set_title(f"Number of New Acceptable States per MAS")
        axs[0, 0].set_ylim([0, None])

        # Number of Tasks from Failure to Acceptable State per MAS
        if not all(x == 0 for x in aggregator.failure_to_acceptable):
            axs[1, 0].scatter(*aggregator._create_histogram(aggregator.failure_to_acceptable), color="black")
            axs[1, 0].set_xlabel("Number of Tasks from Failure to Acceptable State")
            axs[1, 0].set_ylabel("Count of MAS")
            axs[1, 0].set_title(f"Number of Tasks from Failure to Acceptable State per MAS")
            axs[1, 0].set_ylim([0, None])

            print(f"Failure to acceptable: {aggregator._create_histogram(aggregator.failure_to_acceptable)}")
        else:
            axs[1, 0].remove()

        time_axis = [s * (aggregator.evolution_time / aggregator.reliability_steps) for s in
                     range(0, aggregator.reliability_steps)]
        mean_ecm_reliability = numpy.array(aggregator.ecm_mean_task_based_reliabilities).mean(0)
        mean_pcm_reliability = numpy.array(aggregator.pcm_mean_task_based_reliabilities).mean(0)
        ln1 = axs[0, 1].plot(time_axis, list(mean_ecm_reliability), label="ECM-Control")
        ln2 = axs[0, 1].plot(time_axis, list(mean_pcm_reliability), label="PCM-Evolved")

        ax_rif = axs[0, 1].twinx()
        ax_rif.set_ylabel("Improvement Factor")
        reliability_improvement_factors = list(numpy.divide(mean_pcm_reliability, mean_ecm_reliability, out=numpy.zeros_like(mean_pcm_reliability), where=mean_ecm_reliability!=0))
        print(["Time"] + list(time_axis))
        print(["Mean ECM Reliability"] + list(mean_ecm_reliability))
        print(["Mean PCM Reliability"] + list(mean_pcm_reliability))
        print(["Reliability Improvement Factor"] + reliability_improvement_factors)
        print(f"Min Reliability Improvement Factor: {min(reliability_improvement_factors)}, Max: {max(reliability_improvement_factors)}")
        ln3 = ax_rif.plot(time_axis, reliability_improvement_factors, 'k--', label="Improvement Factor")
        ax_rif.set_ylim([0, None])

        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("R(t)")
        axs[0, 1].set_title(f"Mean Task-Based Reliability")

        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        axs[0, 1].legend(lns, labs, loc="lower right")
        axs[0, 1].set_ylim([0, None])

        # Baseline and PCM states
        axs[1, 1].scatter(*aggregator._create_histogram(aggregator.ecm_baseline_states), color="black", label="ECM")
        axs[1, 1].scatter(*aggregator._create_histogram(aggregator.pcm_baseline_states), color="blue", label="PCM")
        axs[1, 1].set_xlabel("Number of States")
        axs[1, 1].set_ylabel("Count of MAS")
        axs[1, 1].set_title(f"Number of States in ECM vs PCM")
        axs[1, 1].set_ylim([0, None])
        axs[1, 1].legend(loc="lower right")

        plt.show()

    """
        Plots the reliability metrics for the given aggregator
    """
    @classmethod
    def plot_reliability_by_task(cls, aggregator):
        if not aggregator.finalized:
            aggregator.finalize()

        fig, axs = plt.subplots(nrows=3, ncols=aggregator.num_tasks)
        fig.suptitle(f"Task-Based Reliability Functions for MASs with ({aggregator.num_agents} agents, {aggregator.num_tasks} tasks)\n")
        fig.tight_layout(pad=1.0)

        for task_index in range(0, aggregator.num_tasks):
            time_axis = [s * (aggregator.evolution_time / aggregator.reliability_steps) for s in range(0, aggregator.reliability_steps)]

            mean_ecm_reliability = numpy.array(aggregator.ecm_task_based_reliabilities[task_index]).mean(0)
            mean_pcm_reliability = numpy.array(aggregator.pcm_task_based_reliabilities[task_index]).mean(0)

            ln1 = axs[0, task_index].plot(time_axis, list(mean_ecm_reliability), label="ECM-Control")
            ln2 = axs[0, task_index].plot(time_axis, list(mean_pcm_reliability), label="PCM-Evolved")
            ax_rif = axs[0, task_index].twinx()
            ax_rif.set_ylabel("Improvement Factor")

            reliability_improvement_factors = list(mean_pcm_reliability / mean_ecm_reliability) if numpy.all(mean_ecm_reliability > 0) else [-1]*aggregator.reliability_steps

            ln3 = ax_rif.plot(time_axis, reliability_improvement_factors, 'k--', label="Improvement Factor")
            ax_rif.set_ylim([0, None])

            axs[0, task_index].set_xlabel("Time (s)")
            axs[0, task_index].set_ylabel("R(t)")
            axs[0, task_index].set_title(f"Task-Based Reliability for Task {task_index+1}/{aggregator.num_tasks}")

            lns = ln1 + ln2 + ln3
            labs = [l.get_label() for l in lns]
            axs[0, task_index].legend(lns, labs, loc="lower left")
            axs[0, task_index].set_ylim([0, None])

            # Plot all reliability functions for the task
            for i in range(0, aggregator.count()):
                ecm_tbrfs = aggregator.ecm_task_based_reliability(i)
                pcm_tbrfs = aggregator.pcm_task_based_reliability(i)
                axs[1, task_index].plot(time_axis, list(numpy.array(ecm_tbrfs[task_index])))
                axs[1, task_index].set_xlabel("Time (s)")
                axs[1, task_index].set_ylabel("R(t)")
                axs[1, task_index].set_title(f"ECM Reliability Functions")

                axs[2, task_index].plot(time_axis, list(numpy.array(pcm_tbrfs[task_index])))
                axs[2, task_index].set_xlabel("Time (s)")
                axs[2, task_index].set_ylabel("R(t)")
                axs[2, task_index].set_title(f"PCM Reliability Functions")

        plt.show()

    """
        Plots a comparison between multiple aggregators
    """
    @classmethod
    def plot_comparison(cls, aggregators, names):
        for agg in aggregators:
            if not agg.finalized:
                agg.finalize()

        fig = plt.figure()
        fig.suptitle(f"Comparison of Reliability Improvement Functions")
        time_axis = [s * (aggregators[0].evolution_time / aggregators[0].reliability_steps) for s in range(0, aggregators[0].reliability_steps)]

        ax = fig.add_subplot(111)

        lns = []
        for i in range(0, len(aggregators)):
            agg = aggregators[i]
            mean_ecm_reliability = numpy.array(agg.ecm_reliability).mean(0)
            mean_pcm_reliability = numpy.array(agg.pcm_reliability).mean(0)
            lns = lns + ax.plot(time_axis, list(mean_pcm_reliability / mean_ecm_reliability) if numpy.all(mean_ecm_reliability > 0) else [1] * agg.reliability_steps, label=f"{names[i]}")

        ax.set_ylabel("Improvement Factor")
        ax.set_ylim([0, None])

        ax.set_xlabel("Time (s)")
        ax.set_title(f"Reliability Improvement Factor During MAS Evolution")

        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc="lower right")
        ax.set_ylim([0, None])

        plt.show()

"""
    Simulation engine for running MAS experiments
"""
class Simulator:
    """
        Thread for printing the progress bar indicating the status of the experiment
    """
    @classmethod
    def _progress_thread(cls, aggregator, number_of_experiments):
        start = time.time()
        while len(aggregator) < number_of_experiments:
            elapsed_time = time.time() - start
            formatted_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            progress_bar_width = 30
            proportion = len(aggregator) / number_of_experiments
            sys.stdout.write(f"\rITM Simulation> [%-{progress_bar_width}s] %d%%     ({formatted_elapsed_time})" % ('=' * int(round(proportion * progress_bar_width, 0)), round(proportion * 100, 2)))
            sys.stdout.flush()

        sys.stdout.write(f"\rITM Simulation done in {formatted_elapsed_time}.\n")
        sys.stdout.flush()

    """
        Generates a MAS and adds it to the aggregator
        args = tuple(aggregator, mas_class, num_agents, num_tasks, number_of_experiments)   
    """
    @classmethod
    def _mas_creator(cls, aggregator, mas_class, num_agents, num_tasks, number_of_experiments):
        while len(aggregator) < number_of_experiments:
            mas = mas_class.create_random_instance(num_agents, num_tasks)
            aggregator.accept_unique(mas)

    """
        Entry point to run a MAS experiment
    """
    @classmethod
    def run(cls, mas_class, number_of_experiments, num_agents, num_tasks, print_progress=True, num_threads=32):
        if print_progress:
            print(f"Running Intelligence Transfer MAS Experiment with {num_agents} agents and {num_tasks} tasks, n={number_of_experiments} ...")

        aggregator = Aggregator(num_agents, num_tasks, max_size=number_of_experiments)

        threads = []
        if print_progress:
            progress_thread = Thread(target=cls._progress_thread, args=(aggregator, number_of_experiments))
            progress_thread.start()
            threads.append(progress_thread)

        for thread_number in range(0, num_threads):
            thread = Thread(target=cls._mas_creator, args=(aggregator, mas_class, num_agents, num_tasks, number_of_experiments))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        aggregator.finalize()
        return aggregator


def main():
    aggregators = []

    ### Run the experiments ###
    aggregators.append(Simulator.run(AcceptableStartMultiAgentSystem, 1000, 3, 3))
    aggregators.append(Simulator.run(UnacceptableStartMultiAgentSystem, 1000, 3, 3))
    aggregators.append(Simulator.run(NoLeafNodeTransferMultiAgentSystem, 1000, 3, 3))
    aggregators.append(Simulator.run(NoIntelligenceTransferMultiAgentSystem, 1000, 3, 3))

    ### Output the aggregated results ###

    for agg in aggregators:
        print(agg)
        Plotter.plot(agg)

    print("a, t, mean ecm states, mean pcm states, mean ecm task redundancy, mean pcm task redundancy, av transitions")
    for agg in aggregators:
        print(f"{agg.num_agents}, {agg.num_tasks}, {statistics.mean(agg.ecm_baseline_states)}, {statistics.mean(agg.pcm_baseline_states)}, {agg.av_ecm_task_redundancy()}, {agg.av_pcm_task_redundancy()}, {agg.av_failure_to_acceptable()}")
#
if __name__ == "__main__":
    main()
