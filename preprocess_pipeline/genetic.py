import random
import numpy as np
from image_operations import *
from metrics import *
import pandas as pd
import cv2
import seaborn as sns

class GeneticAlgorithm:
    def __init__(self, population_size = 120, mutation_rate = 0.08, image_path = None, mask_path = None, metric = None):

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        
        self.image = cv2.imread(image_path)
        self.ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.metric = metric

        self.parameters = {
            'median_filter':{'type':'binary','values':[0,1]},
            'median_filter_size':{'type':'discrete','values':[3,5,7,9]},
            'gaussian_filter':{'type':'binary','values':[0,1]},
            'gaussian_sigma': {'type': 'continuous', 'min': 0, 'max': 1},
            'color_space': {'type': 'discrete', 'values': ['RGB', 'LAB', 'HSV', 'XYZ', 'YCBCR']},
            'channel': {'type': 'discrete', 'values': ['all', 'first', 'second', 'third']},
            'segmentation_method': {'type': 'discrete', 'values': ['clustering', 'pca', 'adaptive', 'otsu']},
            'dilation_size': {'type': 'discrete', 'values': [1, 2, 3, 4]},
            'erosion_size': {'type': 'discrete', 'values': [1, 2, 3, 4]},
            'dilation_size2': {'type': 'discrete', 'values': [1, 2, 3, 4]},
            'erosion_size2': {'type': 'discrete', 'values': [1, 2, 3, 4]}
        }

        self.gene_lengths = {}
        total_length = 0
        for param, config in self.parameters.items():
            if config['type'] == 'binary':
                length = 1
            elif config['type'] == 'discrete':
                length = int(np.ceil(np.log2(len(config['values']))))
            elif config['type'] == 'continuous':
                length = 4  
            self.gene_lengths[param] = length
            total_length += length
        print(f"Total chromosome length: {total_length}")
        self.chromosome_length = total_length
        self.population = self.initialize_population()

    def initialize_population(self):
            return np.random.randint(2, size=(self.population_size, self.chromosome_length))
    

    def decode_chromosome(self, chromosome):
        pos = 0
        params = {}
        
        for param, config in self.parameters.items():
            length = self.gene_lengths[param]
            gene = chromosome[pos:pos+length]
            int_value = int(''.join(map(str, gene)), 2)
            
            if config['type'] == 'binary':
                params[param] = config['values'][int_value % len(config['values'])]
            elif config['type'] == 'discrete':
                max_val = 2**length - 1
                scaled_value = int_value / max_val * (len(config['values']) - 1)
                params[param] = config['values'][round(scaled_value)]
            elif config['type'] == 'continuous':
                max_val = 2**length - 1
                params[param] = config['min'] + (int_value / max_val) * (config['max'] - config['min'])
            
            pos += length
        
        return params
    
    def pipeline(self, chromosome):
        params = self.decode_chromosome(chromosome)
        
        new_img = self.image.copy()
        
        if params['median_filter'] == 1:
            new_img = median_blur(new_img, params['median_filter_size'])
        elif params['gaussian_filter'] == 1:
            new_img = gaussian_blur(new_img, params['gaussian_sigma'])
        new_img = color_space(new_img, params['color_space'])
        new_img = choose_channel(new_img, params['channel'])
        new_img = segmentation(new_img, params['segmentation_method'])
        new_img = dilate(new_img, params['dilation_size'])
        new_img = erode(new_img, params['erosion_size'])
        new_img = dilate(new_img, params['dilation_size2'])
        new_img = erode(new_img, params['erosion_size2'])
        
        return new_img
    
    def selection(self, fitness_scores):
        
        probs = fitness_scores - np.min(fitness_scores)
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones(len(fitness_scores)) / len(fitness_scores)
        
        parents_idx = np.random.choice(len(self.population), size=2, p=probs)
        return self.population[parents_idx[0]], self.population[parents_idx[1]]
    

    def mutation(self, child):
            if np.random.rand() < self.mutation_rate:
                mutate_idx = np.random.randint(len(child))
                child[mutate_idx] = 1 - child[mutate_idx]    
            return child
    
    def crossover(self, parent1, parent2):

        if np.random.rand() < 0.3:
                print("Crossover não realizado (chance menor que 50%)")
                return parent1.copy(), parent2.copy()
 
        size = len(parent1)
        pt1, pt2 = sorted(np.random.choice(size, size=2, replace=False))
        
        print(f"\nPontos de crossover selecionados: {pt1} e {pt2}")
        
        child1 = np.concatenate([parent1[:pt1], parent2[pt1:pt2], parent1[pt2:]])
        child2 = np.concatenate([parent2[:pt1], parent1[pt1:pt2], parent2[pt2:]])

        print(f"Cromossomos pais: {parent1} e {parent2}")
        print(f"Cromossomos filhos: {child1} e {child2}")
        
        return child1, child2
    
    def fitness_function(self, individual):
        image = self.pipeline(individual)
        if self.metric == rand_index:
            ri = self.metric(image, self.ground_truth)
        else:
            ri = 1/(1+self.metric(image, self.ground_truth))
        return ri 

def compare_otsu_pca(image, best_ind_metric, metric, ground_truth, dir):
    otsu_value = []
    pca_value = []
    for i in range(len(image)):
        otsu = segmentation(cv2.imread(dir+image[i]), 'otsu')
        pca = segmentation(cv2.imread(dir+image[i]), 'pca')
        
        otsu_value.append(metric(otsu, cv2.imread(dir+ground_truth[i], cv2.IMREAD_GRAYSCALE)))
        pca_value.append(metric(pca, cv2.imread(dir+ground_truth[i], cv2.IMREAD_GRAYSCALE)))
    
    df = pd.DataFrame({
        'Coluna': image,
        'vet_1': best_ind_metric,
        'vet_2': otsu_value,
        'vet_3': pca_value
    })

        # Criando o gráfico
    plt.figure(figsize=(10, 6))
    indices = range(len(best_ind_metric))

    # Plotando cada vetor com seus respectivos marcadores
    plt.plot(indices, df['vet_1'], 'kx', label='GA11', markersize=10)
    plt.plot(indices, df['vet_2'], 'r*', label='OTSU', markersize=12)
    plt.plot(indices, df['vet_3'], 'bo', label='PCA', markersize=8, fillstyle='none')

    # Configurações do gráfico
    plt.xticks(indices, image)
    plt.xlabel('Colunas')
    plt.ylabel('Valores')
    plt.title('Comparação de Vetores por Coluna')
    plt.grid(alpha=0.3)
    plt.legend()

    # Mostrar o gráfico
    plt.tight_layout()
    plt.savefig("metricas.png")
    plt.show()
        
def main():
    POPULATION_SIZE = 10
    GENERATIONS = 15
    MUTATION_RATE = 0.08
    METRIC = rand_index
    dir_path = "dataset_folhas/"
    mask_paths = ["batata1_mask.png", "batata2_mask.png", "batata3_mask.png", "batata4_mask.png", "pimenta1_mask.png", "pimenta2_mask.png", "pimenta3_mask.png", "pimenta4_mask.png", "tomate1_mask.png", "tomate2_mask.png", "tomate3_mask.png", "tomate4_mask.png"]
    image_paths = ["batata1.JPG", "batata2.JPG", "batata3.JPG", "batata4.JPG", "pimenta1.JPG", "pimenta2.JPG", "pimenta3.JPG", "pimenta4.JPG", "tomate1.JPG", "tomate2.JPG", "tomate3.JPG", "tomate4.JPG"]
    
    best_index_metrics = []
    for ip in range(len(mask_paths)):
        ga = GeneticAlgorithm(population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE, image_path=dir_path+image_paths[ip], mask_path=dir_path+mask_paths[ip], metric = METRIC)
        
        print("="*60)
        print(f"Iniciando Algoritmo Genético")
        print(f"Tamanho da população: {POPULATION_SIZE}")
        print(f"Número de gerações: {GENERATIONS}")
        print(f"Taxa de mutação: {MUTATION_RATE}")
        print("="*60)
        
        
        for generation in range(GENERATIONS):
            print(f"\n{'='*30} Geração {generation+1} {'='*30}")
            
        
            fitness_scores = [ga.fitness_function(ind) for ind in ga.population]
            
            print("\nAvaliação de Fitness:")
            for i, (ind, fit) in enumerate(zip(ga.population, fitness_scores)):
                print(f"Indivíduo {i+1}: {ind} | Fitness: {fit} ({(fit/ga.chromosome_length)*100:.1f}% de 1's)")
            
        
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_individual = ga.population[best_idx]
            
            print(f"\nMelhor da geração: Indivíduo {best_idx+1}")
            print(f"Fitness: {best_fitness} | Cromossomo: {best_individual}")
            
            
            new_population = [best_individual.copy()] 
            
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = ga.selection(fitness_scores)
                
                child1, child2 = ga.crossover(parent1, parent2)
                
                child1 = ga.mutation(child1)
                child2 = ga.mutation(child2)
                
                new_population.extend([child1, child2])
            
            ga.population = np.array(new_population[:POPULATION_SIZE])
            
            avg_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            min_fitness = np.min(fitness_scores)
            
            print(f"\nEstatísticas da geração {generation+1}:")
            print(f"Fitness médio: {avg_fitness:.2f}")
            print(f"Fitness máximo: {max_fitness}")
            print(f"Fitness mínimo: {min_fitness}")
        
        final_fitness = [ga.fitness_function(ind) for ind in ga.population]
        best_idx = np.argmax(final_fitness)
        
        print(ga.decode_chromosome(ga.population[best_idx]))
        print("\n" + "="*60)
        print("Resultado Final:")
        print(f"Melhor indivíduo encontrado: {ga.population[best_idx]}")
        print(f"Fitness do melhor: {final_fitness[best_idx]}")
        print("="*60)

        best_index_metrics.append(final_fitness[best_idx])
        cv2.imwrite(f"masks/{image_paths[ip]}_mask.jpg", ga.pipeline(ga.population[best_idx]))
        cv2.imwrite("Original_Image.jpg", ga.image)
        
    compare_otsu_pca(image_paths, best_index_metrics, METRIC, mask_paths, dir_path)
if __name__ == "__main__":
    main()