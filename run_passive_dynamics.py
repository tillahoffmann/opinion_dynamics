from edge_simulation import *
from simulation import *


class RunExperiments:
    """Class to run experiments from"""

    def __init__(self, num_steps):
        """Initialize the class"""
        # Define a number of nodes and simulation steps
        self.num_nodes = 100
        self.num_steps = num_steps
        # How many balls we start with of each colour
        self.concentration = 3

    def run_many(self):
        """Run many experiemnts!"""
        # To explore passive dynamics
        graph = GraphType(self.num_nodes, 'karateclub')
        mean_belief = self.run_once(graph, 42)
        self.plot_once(stats, mean_belief, "Mean belief per urn")
        # Print out last element of some stats

    def run_once(self, graph, random_seed):
        """Run one experiment"""
        np.random.seed(random_seed)

        # Remove any isolated nodes and relabel the nodes
        graph = remove_isolates(graph)

        # Initialize the nodes
        balls = np.ones((self.num_nodes, 2))

        # Run the simulation
        steps = simulate(graph, balls, self.num_steps)
        mean_belief = evaluate_statistic(balls, steps,
                                         statistic_mean_belief_urn_weighted)

        return mean_belief
       # summary_stats = SummaryStats(alphas, betas, graph)
       # summary_stats.collect_stats()

      #  return summary_stats.stats

    def plot_once(self, stats, property, x_label):
        """Plot a graph for given property"""
        probability = stats[property]
        plt.figure()

        plt.plot(probability)
        plt.xlabel('Step number')
        plt.ylabel(x_label)
        plt.tight_layout()

        plt.show()

if __name__ == '__main__':
    experiment_setup = RunExperiments(10000)
    experiment_setup.run_many()