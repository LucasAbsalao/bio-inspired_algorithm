import random
import numpy as np
from image_operations import *
from metrics import *

class Krill:
    def __init__(self, range_inf, range_sup, z_abs, f,image=None):
        self.f = f
        self.z_abs = z_abs
        self.range_inf = range_inf
        self.range_sup = range_sup
        self.image = image
        self.original_pos = np.array([random.uniform(range_inf, range_sup), random.uniform(range_inf, range_sup)])
        self.fitness = None
        self.best_fitness = None
        self.set_fitness()
        self.Ni = 0
        self.Fi = 0
        self.Di = 0
        self.best_pos = self.original_pos.copy()

        self.parameters = {
            'median_filter':{'type':'binary','values':[0,1]},
            'median_filter_size':{'type':'discrete','values':[3,5,7,9]},
            'gaussian_filter':{'type':'binary','values':[0,1]},
            'gaussian_sigma': {'type': 'continuous', 'min': 0, 'max': 1},
            'segmentation_method': {'type': 'discrete', 'values': ['clustering', 'pca', 'adaptive', 'otsu']},
            'dilation_size': {'type': 'discrete', 'values': [1, 2, 3, 4]},
            'erosion_size': {'type': 'discrete', 'values': [1, 2, 3, 4]},
            'color_space': {'type': 'discrete', 'values': ['RGB', 'NTSC', 'HSV', 'XYZ', 'YCBCR']},
            'channel': {'type': 'discrete', 'values': ['all', 'first', 'second', 'third']},
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
        
    def set_fitness(self, individual):
        return np.sum(individual)
    
    def initialize_population(self):
            return np.random.randint(2, size=(self.population_size, self.chromosome_length))
    

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
        
class Swarm:
    def __init__(self, size, limits, z_abs, f, max_iterations, population_size=10):
        self.krills= []
        self.limits = limits
        self.size = size
        self.f = f
        self.population_size = population_size

        for _ in range(size):
            self.krills.append(Krill(-limits, limits, z_abs, f))
            
        self.kworst = np.inf
        self.kbest = -np.inf
        self.max_iterations = max_iterations
        self.food = 0


    def initialize_population(self):
            return np.random.randint(2, size=(self.population_size, self.chromosome_length))
    
    def evaluate_fitness(self):
        for krill in self.krills:
            krill.set_fitness()
        self.kbest = min(k.fitness for k in self.krills)
        self.kworst = max(k.fitness for k in self.krills)
        
        
    def ds(self, i):
        positions = np.array([k.original_pos for k in self.krills])
        return 1/(5*self.size) * np.sum(np.linalg.norm(self.krills[i].original_pos-positions, axis=1))
    
    def get_neighbours(self, i):
        positions = np.array([k.original_pos for k in self.krills])
        distances = np.linalg.norm(self.krills[i].original_pos - positions, axis=1)
        d = self.ds(i)
        return np.where((distances < d) & (np.arange(self.size) != i))[0]
    
    def xij(self, i_position, positions):
        positions = np.atleast_2d(positions)
        return (positions - i_position)/(np.linalg.norm(positions-i_position, axis=1, keepdims=True) + 1e-12)
    
    def kij(self, i_fitness, fitness):
        return (i_fitness - fitness)/(self.kworst - self.kbest + 1e-12)
    
    def alpha_local(self, i):
        neighbours = self.get_neighbours(i)
        if len(neighbours) == 0:
            return 0
        positions = np.array([self.krills[j].original_pos for j in neighbours])
        fitness = np.array([self.krills[j].fitness for j in neighbours])
        return np.sum(self.kij(self.krills[i].fitness,fitness)[:, np.newaxis] * self.xij(self.krills[i].original_pos, positions))
        
        
    def alpha_target(self, i, iteration):
        cbest = 2 * (random.random() + iteration/self.max_iterations)
        best_ind = np.argmin([krill.fitness for krill in self.krills])
        
        return cbest * self.kij(self.krills[i].fitness, self.krills[best_ind].fitness) * self.xij(self.krills[i].original_pos, self.krills[best_ind].original_pos)
    
    
    def induced_motion(self, i, iteration, Nmax = 0.01, inertia_weight = 0.5):
        alpha_i = self.alpha_local(i) + self.alpha_target(i,iteration)
        Nnew = Nmax*alpha_i + inertia_weight*self.krills[i].Ni
        self.krills[i].Ni = Nnew
        return Nnew
    
    def food_position(self):
        k = np.array([ki.fitness for ki in self.krills])
        x = np.array([xi.original_pos for xi in self.krills])
        self.food = np.sum(x/k[:,np.newaxis], axis=0)/(np.sum(1/k))
        self.food_object.visualize_food(self.food)
    
    def foraging_motion(self, i, iteration, Vf = 0.02, inertia_weight = 0.5):
        beta_i = self.beta_food(i,iteration) + self.beta_best(i)     
        Fnew = Vf*beta_i + inertia_weight*self.krills[i].Fi
        self.krills[i].Fi = Fnew
        return Fnew
    
    def beta_food(self, i, iteration):
        cfood = 2*(1-iteration/self.max_iterations)
        return cfood * self.kij(self.krills[i].fitness, self.f(*self.food)) * self.xij(self.krills[i].original_pos, self.food)
    
    def beta_best(self, i):
        return self.kij(self.krills[i].fitness, self.krills[i].best_fitness) * self.xij(self.krills[i].original_pos, self.krills[i].best_pos)
    
    def physical_diffusion(self, Dmax=0.005):
        self.Di = Dmax * (2 * np.random.rand(2) - 1)
        return Dmax * (2 * np.random.rand(2) - 1)
    
    def change_position(self, i, iteration, Ct=1):
            krill = self.krills[i]
            
            # Calcula os componentes do movimento
            Ni = self.induced_motion(i, iteration).squeeze(0)
            Fi = self.foraging_motion(i, iteration).squeeze(0)
            Di = self.physical_diffusion()
            
            # Soma os componentes para obter a velocidade/direção
            delta = Ni + Fi + Di
            
            # Aplica função sigmoide para transformar em probabilidade
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            
            # Transforma cada componente do delta em probabilidade
            prob_delta = sigmoid(delta) * 2 - 1  # Escala para [-1, 1]
            
            # Calcula o passo temporal
            delta_t = Ct + 0.5*(self.limits - (-self.limits))
            
            # Atualiza a posição usando o delta probabilístico
            krill.original_pos += delta_t * prob_delta
            
            # Garante que a posição está dentro dos limites
            krill.original_pos = np.clip(krill.original_pos, -self.limits, self.limits)
            
            # Atualiza a visualização
            krill.set_pos()
                
            