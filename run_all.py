from helper_functions import *
from control_functions import *
import csv
import sys

#from edge_simulation import *
try:
    from cedge_simulation import *
    seed_rng()
except:
    print "Cython not available. Reverting to python."
    from edge_simulation import *

class RunExperiments:
    """Class to run experiments from"""

    def __init__(self, num_steps, burn_in, burn_out, control_interval,
                 control_balls_fraction, num_runs, num_nodes, graph_types):
        """Initialize the class"""
        # Define a number of nodes and simulation steps
        self.num_steps = num_steps
        self.burn_in = burn_in
        self.burn_out = burn_out
        self.control_interval = control_interval
        self.control_balls_fraction = control_balls_fraction
        self.num_runs = num_runs
        self.num_nodes = num_nodes
        self.graph_types = graph_types

    def run_many(self):
        """Run many experiments!"""
        results = []
        for graph_type in self.graph_types:

            res1 = self.run_one_setup_many_runs(graph_type["type"], graph_type["p"],
                                         control=None)

            res2 = self.run_one_setup_many_runs(graph_type["type"], graph_type["p"],
                                         control=broadcast_control)

            res3 = self.run_one_setup_many_runs(graph_type["type"], graph_type["p"],
                                         control=random_control)

            res4 = self.run_one_setup_many_runs(graph_type["type"], graph_type["p"],
                                         control=hub_control)

            res5 = self.run_one_setup_many_runs(graph_type["type"], graph_type["p"],
                                         control=tom_control)

            res6 = self.run_one_setup_many_runs(graph_type["type"], graph_type["p"],
                                         control=degree_control)

            results += [res1, res2, res3, res4, res5, res6]
        return results

    def run_one_setup_many_runs(self, graph_type, graph_p, control):
        """Run one setup many times"""
        end_props = []
        for num in range(0, self.num_runs):
            """
            from datetime import datetime
            dt = datetime.now()
            running_stats = self.run_once(graph, control)
            end_prop = self.return_end_points(running_stats)
            print (datetime.now() - dt).total_seconds()
            dt = datetime.now()
            """
            graph = GraphType(self.num_nodes, graph_type, p=graph_p)
            end_prop = self.run_once_quick(graph, control)
            end_props.append(end_prop)
            #print (datetime.now() - dt).total_seconds()
        end_prop_distributions = self.collate_end_props(end_props)

        if control is None:
            control_str = "passive"
        else:
            control_str = control.__name__

        if graph_p is not None:
            graph_name = "{}{}".format(graph_type, graph_p)
        else:
            graph_name = graph_type

        mean_results = end_prop_distributions["mean_belief_urn"]
        std_results = end_prop_distributions["std_belief_urns"]
        return [graph_name, control_str] + [np.mean(mean_results)] + \
               [np.std(mean_results)] + [np.mean(std_results)] + \
               [np.std(std_results)]

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
        steps = simulate(graph, balls, self.num_steps, 'steps',
                         control=control, burn_in=self.burn_in,
                         burn_out=self.burn_out,
                         control_balls_fraction=self.control_balls_fraction,
                         control_interval=self.control_interval)

        stats = self.collect_stats(balls, steps)

        return stats

    def run_once_quick(self, graph, control, random_seed=None):
        """Run one experiment"""
        np.random.seed(random_seed)

        # Remove any isolated nodes and relabel the nodes
        graph = remove_isolates(graph)
        graph = graph.to_directed()

        # Initialize the nodes
        balls = np.ones((graph.number_of_nodes(), 2))

        # Run the simulation
        balls = simulate(graph, balls, self.num_steps, 'last',
                         control=control, burn_in=self.burn_in,
                         burn_out=self.burn_out,
                         control_balls_fraction=self.control_balls_fraction,
                         control_interval=self.control_interval)

        stats = self.collect_stats_quick(balls)

        return stats

    def collect_stats(self, balls, steps):
        """Collect stats for the run"""
        stats = {}
        stats["mean_belief_urn"] \
            = evaluate_statistic(balls, steps,
                                 statistic_mean_belief_urn_weighted)
        stats["std_belief_urns"] = \
            evaluate_statistic(balls, steps,
                               statistic_std_belief_urn_weighted)

        return stats

    def collect_stats_quick(self, balls):
        """Collect stats for the run"""
        stats = {}
        stats["mean_belief_urn"] = statistic_mean_belief_urn_weighted(balls)
        stats["std_belief_urns"] = statistic_std_belief_urn_weighted(balls)

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
    # take argument: config file

    def run(input_file, output_file):

        graph_types = [{"type": "erdos", "p": 0.05},
                       {"type": "config", "p": None}]

        output_header = ["graph", "control", "mean of mean belief",
                         "std of mean belief", "mean of std of belief",
                         "std of std of belief"]
        with open(input_file, 'rb') as input:
            with open(output_file, 'wb') as output:
                input_reader = csv.reader(input, delimiter=',')
                output_writer = csv.writer(output, delimiter=',')
                output_writer.writerow(output_header)
                # Skip the header
                next(input_reader, None)
                for line in input_reader:
                    name = line[0]
                    num_steps = int(line[1])
                    burn_in = int(line[2])
                    burn_out = int(line[3])
                    control_interval = int(line[4])
                    control_balls_fraction = float(line[5])
                    num_nodes = int(line[6])
                    num_runs = int(line[7])

                    experiment_setup = \
                        RunExperiments(num_steps, burn_in, burn_out,
                                       control_interval,
                                       control_balls_fraction, num_runs,
                                       num_nodes, graph_types)
                    results = experiment_setup.run_many()
                    output_writer.writerow([name])
                    for result in results:
                        output_writer.writerow(result)

    # needs to be wrapped inside a function to prevent namespace issues in IDEs
    run(sys.argv[1], sys.argv[2])

