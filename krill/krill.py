import random
import vpython as vp
import numpy as np

class Krill:
    def __init__(self, range_inf, range_sup, z_abs, f, scale = 5):
        self.f = f
        self.z_abs = z_abs
        self.scale = scale
        self.range_inf = range_inf
        self.range_sup = range_sup
        self.pos = vp.vector(random.uniform(range_inf, range_sup), random.uniform(range_inf, range_sup), 0)
        self.original_pos = np.array([self.pos.x, self.pos.y])
        self.pos = self.visualize_pos(self.pos)
        self.sphere = vp.sphere(pos = self.pos, color = vp.color.orange, radius = 0.1)
        self.fitness = None
        self.best_fitness = None
        self.set_fitness()
        self.Ni = 0
        self.Fi = 0
        self.Di = 0
        self.best_pos = self.original_pos.copy()
        
    def visualize_pos(self, vector):
        vec = vp.vector(
            self.scale*(-1+2*(vector.x-self.range_inf)/(self.range_sup-self.range_inf)),
            self.scale*(-1+2*(vector.y-self.range_inf)/(self.range_sup-self.range_inf)),
            self.scale*self.f(vector.x, vector.y)/self.z_abs + 0.1
        )
        return vec
        
    def set_fitness(self):
        self.fitness = self.f(self.original_pos[0], self.original_pos[1])
        if self.best_fitness is None:
            self.best_fitness = self.fitness
        if self.fitness < self.best_fitness:
            self.best_fitness = self.fitness
            self.best_pos = self.original_pos

    def set_pos(self):
        x, y = self.original_pos
        z = self.f(x, y)

        # Atualiza valores para visualização
        self.sphere.pos = self.visualize_pos(vp.vector(x,y,z))
    
class Food():
    def __init__(self, range_inf, range_sup, z_abs, f, scale = 5):
        self.range_inf = range_inf
        self.range_sup = range_sup
        self.z_abs = z_abs
        self.f = f
        self.scale = scale
        self.pos = vp.vector(0,0,0)
        self.sphere = vp.sphere(pos = self.pos, radius=0.1, color=vp.color.green)
        
    def visualize_food(self, vector):
        vec = vp.vector(
            self.scale*(-1+2*(vector[0]-self.range_inf)/(self.range_sup-self.range_inf)),
            self.scale*(-1+2*(vector[1]-self.range_inf)/(self.range_sup-self.range_inf)),
            self.scale*self.f(vector[0], vector[1])/self.z_abs
        )
        self.sphere.pos = vec
        
class Swarm:
    def __init__(self, size, limits, z_abs, f, max_iterations, scale = 5):
        self.krills= []
        self.limits = limits
        self.size = size
        self.f = f
        self.scale = scale
        for _ in range(size):
            self.krills.append(Krill(-limits, limits, z_abs, f, scale))
            
        self.kworst = np.inf
        self.kbest = -np.inf
        self.max_iterations = max_iterations
        self.food = 0
        self.food_object = Food(-limits,limits,z_abs,f,scale)
    
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
        return (i_fitness - fitness)/(self.kworst - self.kbest)
    
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
        self.food = np.sum(x/k[:,np.newaxis], axis=0)/np.sum(1/k)
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
        
        Ni = self.induced_motion(i, iteration).squeeze(0)
        Fi = self.foraging_motion(i, iteration).squeeze(0)
        Di = self.physical_diffusion()
        
        
        
        
        delta = Ni + Fi + Di  
        delta_t = Ct + 0.5*(self.limits - (-self.limits))
        
        krill.original_pos += delta_t*delta
        
        krill.set_pos()
        