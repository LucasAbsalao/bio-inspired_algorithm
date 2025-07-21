import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from benchmark_functions import *

# --------- HIPERPARÂMETROS POR FUNÇÃO ---------
FUNCTION_CONFIGS = {
    ackley: {
        'name': 'Ackley',
        'domain': (ACKLEY_MIN_LIMIT, ACKLEY_MAX_LIMIT),
        'pop_size': 80,
        'genes': 30,
        'mutation_prob': 0.01,
        'generations': 500,
        'runs': 30,
        'sigma': 0.1,
        'tournament_k': 3,
        'alpha': 0.5
    },
    rastrigin: {
        'name': 'Rastrigin',
        'domain': (RASTRIGIN_MIN_LIMIT, RASTRIGIN_MAX_LIMIT),
        'pop_size': 50,
        'genes': 30,
        'mutation_prob': 0.01,
        'generations': 700,
        'runs': 30,
        'sigma': 0.05,
        'tournament_k': 5,
        'alpha': 0.6
    },
    schwefel: {
        'name': 'Schwefel',
        'domain': (SCHWEFEL_MIN_LIMIT, SCHWEFEL_MAX_LIMIT),
        'pop_size': 80,
        'genes': 30,
        'mutation_prob': 0.03,
        'generations': 800,
        'runs': 30,
        'sigma': 0.3,
        'tournament_k': 9,
        'alpha': 0.9
    },
    rosenbrock: {
        'name': 'Rosenbrock',
        'domain': (ROSENBROCK_MIN_LIMIT, ROSENBROCK_MAX_LIMIT),
        'pop_size': 100,
        'genes': 30,
        'mutation_prob': 0.05,
        'generations': 1000,
        'runs': 30,
        'sigma': 0.01,
        'tournament_k': 3,
        'alpha': 0.2
    },
}

# --------- FUNÇÕES GENÉTICAS ---------
def generate_population(cfg):
    dmin, dmax = cfg['domain']
    return np.random.uniform(dmin, dmax, (cfg['pop_size'], cfg['genes']))

def fitness(pop, f):
    return np.apply_along_axis(f, 1, pop)

def tournament_selection(pop, fitness_scores, cfg):
    size = cfg['pop_size']
    k = cfg['tournament_k']
    idxs = np.random.randint(0, size, size=(size, k))
    selected = np.argmin(fitness_scores[idxs], axis=1)
    return pop[idxs[np.arange(size), selected]]

def crossover_blend(p1, p2, alpha):
    diff = np.abs(p1 - p2)
    min_vals = np.minimum(p1, p2) - alpha * diff
    max_vals = np.maximum(p1, p2) + alpha * diff
    return np.random.uniform(min_vals, max_vals)

def apply_crossover(parents, cfg):
    dmin, dmax = cfg['domain']
    size = cfg['pop_size']
    alpha = cfg['alpha']
    np.random.shuffle(parents)
    offspring = np.empty_like(parents)
    for i in range(0, size, 2):
        if i+1 < size:
            c1 = np.clip(crossover_blend(parents[i], parents[i+1], alpha), dmin, dmax)
            c2 = np.clip(crossover_blend(parents[i+1], parents[i], alpha), dmin, dmax)
            offspring[i], offspring[i+1] = c1, c2
    return offspring

def mutate_gaussian(pop, cfg):
    dmin, dmax = cfg['domain']
    sigma = cfg['sigma'] * (dmax - dmin)
    mutation_prob = cfg['mutation_prob']
    mutation_mask = np.random.rand(*pop.shape) < mutation_prob
    mutations = np.random.normal(0, sigma, size=pop.shape)
    mutated_pop = pop + mutation_mask * mutations
    return np.clip(mutated_pop, dmin, dmax)

# --------- ALGORITMO GENÉTICO ---------
def genetic_algorithm(f, cfg):
    pop = generate_population(cfg)
    fit = fitness(pop, f)
    best_idx = np.argmin(fit)
    best_ind = pop[best_idx].copy()
    best_fit = fit[best_idx]
    history = []

    for _ in range(cfg['generations']):
        parents = tournament_selection(pop, fit, cfg)
        offspring = apply_crossover(parents, cfg)
        offspring = mutate_gaussian(offspring, cfg)
        offspring_fit = fitness(offspring, f)

        # elitismo
        worst_idx = np.argmax(offspring_fit)
        if best_fit < offspring_fit[worst_idx]:
            offspring[worst_idx] = best_ind
            offspring_fit[worst_idx] = best_fit

        pop = offspring
        fit = offspring_fit

        cur_best_idx = np.argmin(fit)
        if fit[cur_best_idx] < best_fit:
            best_fit = fit[cur_best_idx]
            best_ind = pop[cur_best_idx].copy()

        history.append(best_fit)

    return best_ind, best_fit, history

# --------- EXECUÇÃO E VISUALIZAÇÃO ---------
def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sns.set_theme(style="whitegrid")
    axes = axes.flatten()

    for i, (func, cfg) in enumerate([(rastrigin, FUNCTION_CONFIGS[rastrigin])]):
        print(f"=== Otimizando função {cfg['name']} ===")

        global_best_val = float('inf')
        global_best = None
        run_history = []
        runs = []

        for _ in range(cfg['runs']):
            best_ind, val, history = genetic_algorithm(func, cfg)
            runs.append(val)
            run_history.append(history)

            if val < global_best_val:
                global_best = best_ind
                global_best_val = val

        average = np.mean(runs)
        deviation = np.std(runs)
        gens = np.arange(cfg['generations'])

        run_history = np.array(run_history)
        mean_curve = np.mean(run_history, axis=0)
        std_curve = np.std(run_history, axis=0)

        ax = axes[i]
        ax.plot(gens, mean_curve, label="Média", color="blue")
        ax.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve,
                        color="blue", alpha=0.3, label="Desvio Padrão")
        ax.set_title(f"Convergência - {cfg['name']}")
        ax.set_xlabel("Geração")
        ax.set_ylabel("Valor")
        ax.legend(loc="upper right", fontsize=9)

        stats_text = (
            "Valor Final\n"
            f"µ: {average:.6f}\n"
            f"σ: {deviation:.6f}\n"
            f"Mínimo: {global_best_val:.6f}"
        )
        ax.text(0.98, 0.85, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
