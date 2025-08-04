import random
import numpy as np
from image_operations import *
from metrics import *

class Krill:
    def __init__(self, range_inf, range_sup, image=None, ground_truth = None, metric = None):
        self.range_inf = range_inf
        self.range_sup = range_sup
        self.image = image
        self.ground_truth = ground_truth
        self.metric = metric
        #self.original_pos = np.array([random.uniform(range_inf, range_sup), random.uniform(range_inf, range_sup)])
        self.gene_lengths = {}

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
        #print(f"Total chromosome length: {total_length}")
        self.chromosome_length = total_length
        self.chromosome = np.random.uniform(low=-1, high=1, size=self.chromosome_length)
        self.binary = None
        self.get_binary()

        self.fitness = None
        self.best_fitness = None

        self.set_fitness()
        self.Ni = 0
        self.Fi = 0
        self.Di = 0
        self.best_chromosome = self.chromosome.copy()


    def __str__(self):
        return str(self.chromosome)

    def decode_chromosome(self):
        pos = 0
        params = {}
        
        for param, config in self.parameters.items():
            length = self.gene_lengths[param]
            gene = self.binary[pos:pos+length]
            #print(self.binary)
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
        
    def set_fitness(self):

        image = self.pipeline()
        if self.metric == rand_index:
            ri = self.metric(image, self.ground_truth)
        else:
            ri = 1/(1+self.metric(image, self.ground_truth))
        self.fitness = ri
        #print(self.fitness)
        if self.best_fitness is None:
            self.best_fitness = self.fitness
        if self.fitness > self.best_fitness:
            self.best_fitness = self.fitness
            self.best_chromsome = self.chromosome

    def pipeline(self):
        params = self.decode_chromosome()
        #print(params)
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
    
    def get_binary(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        prob_delta = sigmoid(self.chromosome)
        self.binary = (np.random.uniform(size=self.chromosome.shape) > prob_delta).astype(int)


class Swarm:
    def __init__(self, size, limits, max_iterations, population_size=10, image=None, image_path=None, mask_path=None, metric = None):
        self.selfs= []
        self.limits = limits
        self.size = size
        self.population_size = population_size

        self.image = cv2.imread(image_path)
        self.ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.metric = metric

        self.krills = []
        for _ in range(size):
            self.krills.append(Krill(-limits, limits, self.image, self.ground_truth, self.metric))
            
        self.kworst = np.inf
        self.kbest = -np.inf
        self.max_iterations = max_iterations
        self.food = 0
    
    def evaluate_fitness(self):
        for krill in self.krills:
            krill.set_fitness()
        self.kbest = max(k.fitness for k in self.krills)
        self.kworst = min(k.fitness for k in self.krills)
        
        
    def ds(self, i):
        positions = np.array([k.chromosome for k in self.krills])
        return 1/(5*self.size) * np.sum(np.linalg.norm(self.krills[i].chromosome-positions, axis=1))
    
    def get_neighbours(self, i):
        positions = np.array([k.chromosome for k in self.krills])
        distances = np.linalg.norm(self.krills[i].chromosome - positions, axis=1)
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
        positions = np.array([self.krills[j].chromosome for j in neighbours])
        fitness = np.array([self.krills[j].fitness for j in neighbours])
        return np.sum(self.kij(self.krills[i].fitness,fitness)[:, np.newaxis] * self.xij(self.krills[i].chromosome, positions))
        
        
    def alpha_target(self, i, iteration):
        cbest = 2 * (random.random() + iteration/self.max_iterations)
        best_ind = np.argmin([krill.fitness for krill in self.krills])
        
        return cbest * self.kij(self.krills[i].fitness, self.krills[best_ind].fitness) * self.xij(self.krills[i].chromosome, self.krills[best_ind].chromosome)
    
    
    def induced_motion(self, i, iteration, Nmax = 0.01, inertia_weight = 0.5):
        alpha_i = self.alpha_local(i) + self.alpha_target(i,iteration)
        Nnew = Nmax*alpha_i + inertia_weight*self.krills[i].Ni
        self.krills[i].Ni = Nnew
        return Nnew
    
    def food_position(self):
        k = np.array([ki.fitness for ki in self.krills])
        x = np.array([xi.chromosome for xi in self.krills])
        self.food = np.sum(x/k[:,np.newaxis], axis=0)/(np.sum(1/k))
        #self.food_object.visualize_food(self.food)
    
    def foraging_motion(self, i, iteration, Vf = 0.02, inertia_weight = 0.5):
        beta_i = self.beta_food(i,iteration) + self.beta_best(i)     
        Fnew = Vf*beta_i + inertia_weight*self.krills[i].Fi
        self.krills[i].Fi = Fnew
        return Fnew
    
    def beta_food(self, i, iteration):
        cfood = 2*(1-iteration/self.max_iterations)
        dummy = Krill(-self.limits, self.limits, self.image, self.ground_truth, self.metric)
        dummy.chromosome = self.food
        dummy.get_binary()
        segmented = dummy.pipeline()
    
        similarity = self.metric(segmented, self.ground_truth)
        return cfood * self.kij(self.krills[i].fitness, similarity) * self.xij(self.krills[i].chromosome, self.food)
    
    def beta_best(self, i):
        return self.kij(self.krills[i].fitness, self.krills[i].best_fitness) * self.xij(self.krills[i].chromosome, self.krills[i].best_chromosome)
    
    def physical_diffusion(self, Dmax=0.005):
        self.Di = Dmax * (2 * np.random.rand(self.krills[0].chromosome_length) - 1)
        return Dmax * (2 * np.random.rand(self.krills[0].chromosome_length) - 1)
    
    def change_position(self, i, iteration, Ct=1):
        krill = self.krills[i]
        
        Ni = self.induced_motion(i, iteration).squeeze(0)
        Fi = self.foraging_motion(i, iteration).squeeze(0)
        Di = self.physical_diffusion()
        
        delta = Ni + Fi + Di
        delta_t = Ct + 0.5*(self.limits - (-self.limits)-1)
        krill.chromosome += delta_t*delta

        krill.get_binary()
        
            

if __name__ == "__main__":
    METRIC = rand_index
    dir_path = "dataset_folhas/"
    mask_paths = ["batata1_mask.png", "batata2_mask.png", "batata3_mask.png", "batata4_mask.png",
                "pimenta1_mask.png", "pimenta2_mask.png", "pimenta3_mask.png", "pimenta4_mask.png",
                "tomate1_mask.png", "tomate2_mask.png", "tomate3_mask.png", "tomate4_mask.png"]
    image_paths = ["batata1.JPG", "batata2.JPG", "batata3.JPG", "batata4.JPG",
                "pimenta1.JPG", "pimenta2.JPG", "pimenta3.JPG", "pimenta4.JPG",
                "tomate1.JPG", "tomate2.JPG", "tomate3.JPG", "tomate4.JPG"]
    limits = 1
    n_iteracoes = 10

    # Lista para armazenar os melhores indivíduos de cada imagem
    all_best_individuals = []

    for ip in range(len(mask_paths) - 10):
        swarm = Swarm(15, limits, n_iteracoes, population_size=10,
                    image_path=dir_path + image_paths[ip],
                    mask_path=dir_path + mask_paths[ip],
                    metric=METRIC)

        # Vetor para guardar o melhor fitness de cada iteração (opcional)
        best_fitness_per_iteration = []

        x = 0
        while x < n_iteracoes:
            swarm.evaluate_fitness()
            swarm.food_position()

            for i in range(swarm.size):
                swarm.change_position(i, x, 0.1)

            # Coleta do melhor indivíduo da iteração
            best_krill = max(swarm.krills, key=lambda k: k.fitness)
            best_fitness_per_iteration.append(best_krill.fitness)

            x += 1

        # Armazena o melhor indivíduo da última iteração
        best_overall = max(swarm.krills, key=lambda k: k.fitness)
        all_best_individuals.append(best_overall)

        print(f"Melhor fitness da imagem {image_paths[ip]}: {best_overall.fitness}")