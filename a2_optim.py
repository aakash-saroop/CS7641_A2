import numpy as np
import six
import sys
sys.modules['sklearn.externals.six'] = six
import logging
import matplotlib.pyplot as plt
import time
import mlrose

# Setup logging
logging.basicConfig(level=logging.INFO)

# Parameters
target_string = np.random.randint(0, 2, 30)
bit_length = 30
initial_state = np.random.randint(0, 2, bit_length)
max_iterations = 1000
initial_temp = 10
cooling_rate = 0.99
population_size = 200
generations = 100
mutation_rate = 0.1


def execute_algorithms():
    results = {
        'RHC_SA': {'iterations': [], 'fitness': [], 'time': 0},
        'SA': {'iterations': [], 'fitness': [], 'time': 0},
        'GA_SA': {'iterations': [], 'fitness': [], 'time': 0},
        'RHC_GA': {'iterations': [], 'fitness': [], 'time': 0},
        'SA_GA': {'iterations': [], 'fitness': [], 'time': 0},
        'GA_GA': {'iterations': [], 'fitness': [], 'time': 0},
        'MIMIC_GA': {'iterations': [], 'fitness': [], 'time': 0}  # Added MIMIC for GA
    }

    # Define fitness functions for mlrose problems
    def fitness_sa(state):
        score = np.sum(state)
        penalty = np.sum([1 for i in range(len(state) - 1) if state[i] == 1 and state[i + 1] == 1])
        return score - penalty

    def fitness_ga(state):
        return np.sum(state == target_string)

    # Define mlrose optimization problems
    problem_sa = mlrose.DiscreteOpt(length=bit_length, fitness_fn=mlrose.CustomFitness(fitness_sa), maximize=True, max_val=2)
    problem_ga = mlrose.DiscreteOpt(length=bit_length, fitness_fn=mlrose.CustomFitness(fitness_ga), maximize=True, max_val=2)

    # Randomized Hill Climbing for SA
    start_time = time.time()
    best_state, best_fitness, curve = mlrose.random_hill_climb(problem_sa, max_attempts=100, max_iters=max_iterations,
                                                        restarts=20, init_state=initial_state, curve=True)
    end_time = time.time()
    results['RHC_SA']['iterations'] = list(range(len(curve)))
    results['RHC_SA']['fitness'] = curve
    results['RHC_SA']['time'] = end_time - start_time

    print("RHC_SA:")
    print("Iterations:", results['RHC_SA']['iterations'])
    print("Fitness:", results['RHC_SA']['fitness'])
    print("Time:", results['RHC_SA']['time'])

    # Simulated Annealing
    start_time = time.time()
    best_state, best_fitness, curve = mlrose.simulated_annealing(problem_sa, schedule=mlrose.GeomDecay(init_temp=initial_temp, decay=cooling_rate),
                                                          max_attempts=100, max_iters=max_iterations, init_state=initial_state, curve=True)
    end_time = time.time()
    results['SA']['iterations'] = list(range(len(curve)))
    results['SA']['fitness'] = curve
    results['SA']['time'] = end_time - start_time

    print("\nSA:")
    print("Iterations:", results['SA']['iterations'])
    print("Fitness:", results['SA']['fitness'])
    print("Time:", results['SA']['time'])

    # Genetic Algorithm for SA
    start_time = time.time()
    best_state, best_fitness, curve = mlrose.genetic_alg(problem_sa, pop_size=population_size, max_attempts=100,
                                                  max_iters=generations, mutation_prob=mutation_rate, curve=True)
    end_time = time.time()
    results['GA_SA']['iterations'] = list(range(len(curve)))
    results['GA_SA']['fitness'] = curve
    results['GA_SA']['time'] = end_time - start_time

    print("\nGA_SA:")
    print("Iterations:", results['GA_SA']['iterations'])
    print("Fitness:", results['GA_SA']['fitness'])
    print("Time:", results['GA_SA']['time'])

    # Randomized Hill Climbing for GA
    start_time = time.time()
    best_state, best_fitness, curve = mlrose.random_hill_climb(problem_ga, max_attempts=100, max_iters=max_iterations,
                                                        restarts=20, init_state=initial_state, curve=True)
    end_time = time.time()
    results['RHC_GA']['iterations'] = list(range(len(curve)))
    results['RHC_GA']['fitness'] = curve
    results['RHC_GA']['time'] = end_time - start_time

    print("\nRHC_GA:")
    print("Iterations:", results['RHC_GA']['iterations'])
    print("Fitness:", results['RHC_GA']['fitness'])
    print("Time:", results['RHC_GA']['time'])

    # Simulated Annealing for GA
    start_time = time.time()
    best_state, best_fitness, curve = mlrose.simulated_annealing(problem_ga, schedule=mlrose.GeomDecay(init_temp=initial_temp, decay=cooling_rate),
                                                          max_attempts=100, max_iters=max_iterations, init_state=initial_state, curve=True)
    end_time = time.time()
    results['SA_GA']['iterations'] = list(range(len(curve)))
    results['SA_GA']['fitness'] = curve
    results['SA_GA']['time'] = end_time - start_time

    print("\nSA_GA:")
    print("Iterations:", results['SA_GA']['iterations'])
    print("Fitness:", results['SA_GA']['fitness'])
    print("Time:", results['SA_GA']['time'])

    # Genetic Algorithm for GA
    start_time = time.time()
    best_state, best_fitness, curve = mlrose.genetic_alg(problem_ga, pop_size=population_size, max_attempts=100,
                                                  max_iters=generations, mutation_prob=mutation_rate, curve=True)
    end_time = time.time()
    results['GA_GA']['iterations'] = list(range(len(curve)))
    results['GA_GA']['fitness'] = curve
    results['GA_GA']['time'] = end_time - start_time

    print("\nGA_GA:")
    print("Iterations:", results['GA_GA']['iterations'])
    print("Fitness:", results['GA_GA']['fitness'])
    print("Time:", results['GA_GA']['time'])

    # MIMIC for GA
    start_time = time.time()
    best_state, best_fitness, curve = mlrose.mimic(problem_ga, pop_size=population_size, max_attempts=100,
                                                   max_iters=generations, keep_pct=0.2, curve=True)
    end_time = time.time()
    results['MIMIC_GA']['iterations'] = list(range(len(curve)))
    results['MIMIC_GA']['fitness'] = curve
    results['MIMIC_GA']['time'] = end_time - start_time

    print("\nMIMIC_GA:")
    print("Iterations:", results['MIMIC_GA']['iterations'])
    print("Fitness:", results['MIMIC_GA']['fitness'])
    print("Time:", results['MIMIC_GA']['time'])

    return results


# Plot functions remain unchanged

def plot_fitness_iteration(results):
    for label, data in results.items():
        plt.plot(data['iterations'], data['fitness'], label=label)
    plt.title('Fitness over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()


def plot_fitness_problem_size(results, problem_sizes):
    labels = list(results.keys())
    for label in labels:
        fitnesses = []
        for size in problem_sizes:
            fitnesses.append(np.mean(results[label]['fitness'][:size]))  # Calculate mean fitness for each problem size
        plt.plot(problem_sizes, fitnesses, label=label)
    plt.title('Fitness over Problem Sizes')
    plt.xlabel('Problem Size')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()


def plot_function_evaluations(results):
    for label, data in results.items():
        plt.plot(data['iterations'], data['fitness'], label=label)
    plt.title('Fitness over Function Evaluations')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()


def plot_wall_clock_time(results):
    labels = results.keys()
    times = [data['time'] for data in results.values()]
    plt.bar(labels, times)
    plt.title('Wall Clock Time')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (seconds)')
    plt.show()

def plot_ga_vs_mimic(results):
    ga_data = results['GA_GA']
    mimic_data = results['MIMIC_GA']
    
    plt.plot(ga_data['iterations'], ga_data['fitness'], label='Genetic Algorithm')
    plt.plot(mimic_data['iterations'], mimic_data['fitness'], label='MIMIC')
    plt.title('GA vs MIMIC: Fitness over Iterations (GA Problem)')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend()
    plt.show()

# Execute algorithms and collect results
results = execute_algorithms()

# Generate plots
plot_fitness_iteration(results)
plot_fitness_problem_size(results, [10, 20, 30])
plot_function_evaluations(results)
plot_wall_clock_time(results)

# Generate specific comparison plot
plot_ga_vs_mimic(results)
