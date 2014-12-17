from edge_simulation import *
from helper_functions import *


class RunExperiments:
    """Class to run experiments from"""

    def __init__(self, num_steps, num_runs):
        """Initialize the class"""
        # Define a number of nodes and simulation steps
        self.num_nodes = 100
        self.num_steps = num_steps
        self.num_runs = num_runs

    def run_many(self):
        """Run many experiments!"""
        # To explore passive dynamics
        end_props = []
        graph = GraphType(self.num_nodes, 'erdos')

        # Show the time-course of one run -- useful to check if we have
        # settled down
        stats = self.run_once(graph, None)
        #self.plot_once(stats["mean_belief_urn"], "mean belief of urns")
        #self.plot_once(stats["mean_belief_balls"], "mean belief overall")

        # Show the distribution of end values
        for num in range(0, self.num_runs):
            running_stats = self.run_once(graph, None)
            end_prop = self.return_end_points(running_stats)
            end_props.append(end_prop)
        end_prop_distributions = self.collate_end_props(end_props)

        self.plot_hist(end_prop_distributions["mean_belief_urn"],
                       "mean urn end belief")
        self.plot_hist(end_prop_distributions["mean_belief_balls"],
                       "mean overall end belief")

    def return_end_points(self, running_stats):
        """Calculate statistics at end point"""
        end_prop = {}
        for property in running_stats.keys():
            end_prop[property] = running_stats[property][self.num_steps - 1]

        return end_prop

    def collate_end_props(self, end_props):
        """Take array of dict, and return dict of arrays"""
        # Initialize a dict of empty lists
        end_prop_distributions = {}
        properties = end_props[0].keys()
        for property in properties:
            end_prop_distributions[property] = []

        # Populate this dict
        for simulation_end_point in end_props:
            for property in properties:
                end_prop_distributions[property]\
                    .append(simulation_end_point[property])

        return end_prop_distributions

    def run_once(self, graph, random_seed):
        """Run one experiment"""
        np.random.seed(random_seed)

        # Remove any isolated nodes and relabel the nodes
        graph = remove_isolates(graph)

        # Initialize the nodes
        balls = np.ones((self.num_nodes, 2))

        # Run the simulation
        steps = simulate(graph, balls, self.num_steps)
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

        #TODO: add more summary stats here
        return stats

    def plot_once(self, property, ylabel):
        """Plot a graph for given property"""

        plt.figure()
        plt.plot(property)
        plt.xlabel('Step number')
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    def plot_hist(self, property, xlabel):
        """Plot a histogram for given property"""

        plt.figure()
        plt.hist(property)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Initialize with num timesteps, num runs
     experiment_setup = RunExperiments(1000, 50)
     experiment_setup.run_many()