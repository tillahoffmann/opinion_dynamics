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

            res1 = self.run_one_setup_many_runs(graph_type["type"],
                                                graph_type["p"],
                                                control=None)

            res2 = self.run_one_setup_many_runs(graph_type["type"],
                                                graph_type["p"],
                                                control=broadcast_control)

            res3 = self.run_one_setup_many_runs(graph_type["type"],
                                                graph_type["p"],
                                                control=random_control)

            res4 = self.run_one_setup_many_runs(graph_type["type"],
                                                graph_type["p"],
                                                control=hub_control)

            res5 = self.run_one_setup_many_runs(graph_type["type"],
                                                graph_type["p"],
                                                control=tom_control)

            res6 = self.run_one_setup_many_runs(graph_type["type"],
                                                graph_type["p"],
                                                control=degree_control)

            res7 = self.run_one_setup_many_runs(graph_type["type"],
                                                graph_type["p"],
                                                control=influence_control)

            res8 = self.run_one_setup_many_runs(graph_type["type"],
                                                graph_type["p"],
                                                control=young_control)

            results += [res1, res2, res3, res4, res5, res6, res7, res8]
        return results


    def run_one_setup_many_runs(self, graph_type, graph_p, control):
        """Run one setup many times"""
        means = []
        stds = []
        for num in range(0, self.num_runs):
            graph = GraphType(self.num_nodes, graph_type, p=graph_p)
            mean, std = self.run_once(graph, control)
            means.append(mean)
            stds.append(std)

        if control is None:
            control_str = "passive"
        else:
            control_str = control.__name__

        if graph_p is not None:
            graph_name = "{}{}".format(graph_type, graph_p)
        else:
            graph_name = graph_type

        return [graph_name, control_str] + [np.mean(means)] + \
               [np.std(means)] + [np.mean(stds)] + \
               [np.std(stds)]

    def run_once(self, graph, control, random_seed=None):
        """Run one experiment"""
        np.random.seed(random_seed)

        # Remove any isolated nodes and relabel the nodes
        graph = remove_isolates(graph)
        graph = graph.to_directed()

        # Initialize the nodes
        balls = np.ones((graph.number_of_nodes(), 2))

        # Run the simulation
        balls = simulate(graph, balls, self.num_steps, output='last',
                         control=control, burn_in=self.burn_in,
                         burn_out=self.burn_out,
                         control_balls_fraction=self.control_balls_fraction,
                         control_interval=self.control_interval)

        mean = statistic_mean_belief_urn_weighted(balls)
        std = statistic_std_belief_urn_weighted(balls)

        return mean, std

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


def run(input_file, output_file):
    """Run all experiments"""

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


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])

