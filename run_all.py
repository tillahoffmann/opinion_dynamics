from edge_simulation import *
from helper_functions import *
from control_functions import *
import csv


class RunExperiments:
    """Class to run experiments from"""

    def __init__(self, num_steps, num_runs, graphs, graph_names,
                 std_file_name, mean_file_name):
        """Initialize the class"""
        # Define a number of nodes and simulation steps
        self.num_steps = num_steps
        self.num_runs = num_runs
        self.graphs = graphs
        self.graph_names = graph_names
        self.std_file_name = std_file_name
        self.mean_file_name = mean_file_name

    def run_many(self):
        """Run many experiments!"""
        # To explore passive dynamics
        with open(self.mean_file_name, 'wb') as mean_csvfile:
            with open(self.std_file_name, 'wb') as std_csvfile:
                mean_results_writer = csv.writer(mean_csvfile, delimiter=',')
                std_results_writer = csv.writer(std_csvfile, delimiter=',')
                for idx, graph in enumerate(self.graphs):
                    print(self.graph_names[idx])
                    # Show the time-course of one run -- useful to check if we have
                    # settled down
                    # stats = self.run_once(graph, control=None)
                    # self.plot_once(stats["mean_belief_urn"], "mean belief of urns")

                    self.run_one_setup_many_runs(graph, self.graph_names[idx],
                                                 mean_results_writer,
                                                 std_results_writer,
                                                 control=None)

                    self.run_one_setup_many_runs(graph, self.graph_names[idx],
                                                 mean_results_writer,
                                                 std_results_writer,
                                                 control=broadcast_control)

                    self.run_one_setup_many_runs(graph, self.graph_names[idx],
                                                 mean_results_writer,
                                                 std_results_writer,
                                                 control=random_control)

                    self.run_one_setup_many_runs(graph, self.graph_names[idx],
                                                 mean_results_writer,
                                                 std_results_writer,
                                                 control=hub_control)

                    self.run_one_setup_many_runs(graph, self.graph_names[idx],
                                                 mean_results_writer,
                                                 std_results_writer,
                                                 control=tom_control)

    def run_one_setup_many_runs(self, graph, graph_name, mean_results_writer,
                                std_results_writer, control):
        """Run one setup many times"""
        end_props = []
        for num in range(0, self.num_runs):
            running_stats = self.run_once(graph, control)
            end_prop = self.return_end_points(running_stats)
            end_props.append(end_prop)
        end_prop_distributions = self.collate_end_props(end_props)

        if control is None:
            control_str = "passive"
        else:
            control_str = control.__name__

        mean_urns = [graph_name, control_str] + end_prop_distributions["mean_belief_urn"]
        std_urns = [graph_name, control_str] + end_prop_distributions["std_belief_urns"]

        mean_results_writer.writerow(mean_urns)
        std_results_writer.writerow(std_urns)

        # self.plot_hist(end_prop_distributions["mean_belief_urn"], "mean urn end belief")

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

    def run_once(self, graph, control, random_seed=None):
        """Run one experiment"""
        np.random.seed(random_seed)

        # Remove any isolated nodes and relabel the nodes
        graph = remove_isolates(graph)
        graph = graph.to_directed()

        # Initialize the nodes
        balls = np.ones((graph.number_of_nodes(), 2))

        # Run the simulation
        steps = simulate(graph, balls, self.num_steps, control=control)
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
        stats["std_belief_urns"] = \
            evaluate_statistic(balls, steps,
                               statistic_std_belief_urn_weighted)

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
    num_nodes = 100
    num_steps = 10000
    num_runs = 50
    graph_ER_01 = GraphType(num_nodes, 'erdos', p=0.01)
    graph_ER_05 = GraphType(num_nodes, 'erdos', p=0.05)
    graph_ER_1 = GraphType(num_nodes, 'erdos', p=0.1)
    graph_BA_1 = GraphType(num_nodes, 'powerlaw', m=1, p=0)
    graph_BA_3 = GraphType(num_nodes, 'powerlaw', m=3, p=0)
    graph_BA_5 = GraphType(num_nodes, 'powerlaw', m=5, p=0)
    graphs = [graph_ER_01, graph_ER_05, graph_ER_1,
              graph_BA_1, graph_BA_3, graph_BA_5]
    graph_names = ["graph_ER_01", "graph_ER_05", "graph_ER_1", "graph_BA_1",
                   "graph_BA_3", "graph_BA_5"]
    experiment_setup = \
        RunExperiments(num_steps, num_runs, graphs, graph_names,
                       "std_belief_urns_50.csv", "mean_belief_urns_50.csv")
    experiment_setup.run_many()