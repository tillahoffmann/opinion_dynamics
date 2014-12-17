from edge_simulation import *
from edge_simulation import simulate as till_simulate
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
        """Run many experiments!"""
        # To explore passive dynamics
        endproperties = []
        graph = GraphType(self.num_nodes, 'erdos')
        for i in range(1,1000):
            stats = self.run_once(graph, None)
            endproperties.append(stats["mean_belief_urn"][len(stats["mean_belief_urn"])-1])
            #self.plot_once(stats["mean_belief_urn"], "mean belief of urns")
            #self.plot_once(stats["mean_belief_balls"], "mean belief balls")
        self.plot_once(endproperties,"mean end belief")


    def run_once(self, graph, random_seed):
        """Run one experiment"""
        np.random.seed(random_seed)

        # Remove any isolated nodes and relabel the nodes
        graph = remove_isolates(graph)

        # Initialize the nodes
        balls = np.ones((self.num_nodes, 2))

        # Run the simulation
        steps = till_simulate(graph, balls, self.num_steps)
        stats = self.collect_stats(balls, steps)

        return stats

    def collect_stats(self, balls, steps):
        """Collect stats for the run"""
        stats = {}
        stats["mean_belief_urn"] \
            = evaluate_statistic(balls, steps,
                                 statistic_mean_belief_urn_weighted)
        stats["mean_belief_balls"] = \
            evaluate_statistic(balls, steps,
                               statistic_mean_belief_ball_weighted)

        return stats

    def plot_once(self, property, ylabel):
        """Plot a graph for given property"""

        plt.figure()

        plt.hist(property)
        plt.xlabel('Step number')
        plt.ylabel(ylabel)
        plt.tight_layout()

        plt.show()

if __name__ == '__main__':
     experiment_setup = RunExperiments(1000)
     experiment_setup.run_many()