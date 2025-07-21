import numpy as np

ACKLEY_MAX_LIMIT = 32.768
ACKLEY_MIN_LIMIT = -32.768
ACKLEY_GLOBAL_MINIMUM_PARAM = 0

RASTRIGIN_MAX_LIMIT = 5.12
RASTRIGIN_MIN_LIMIT = -5.12
RASTRIGIN_GLOBAL_MINIMUM_PARAM = 0

SCHWEFEL_MAX_LIMIT = 500
SCHWEFEL_MIN_LIMIT = -500
SCHWEFEL_GLOBAL_MINIMUM_PARAM = 420.9687

ROSENBROCK_MAX_LIMIT = 10
ROSENBROCK_MIN_LIMIT = -5
ROSENBROCK_GLOBAL_MINIMUM_PARAM = 1

def ackley(vector:np.ndarray, dimensions:int=None, a:float=20, b:float=0.2, c:float=2*np.pi):
    if dimensions is not None and dimensions != vector.shape[0]:
        raise Exception("Number of dimensions is different from the vector shape")
    d = vector.shape[0]
    
    return -a * np.exp(-b*np.sqrt((1/d)*np.sum(vector**2))) - np.exp((1/d)*np.sum(np.cos(c*vector))) + a + np.exp(1)

def rastrigin(vector:np.ndarray, dimensions:int=None):
    if dimensions is not None and dimensions != vector.shape[0]:
        raise Exception("Number of dimensions is different from the vector shape")
    d = vector.shape[0]
    
    return 10*d + np.sum(vector**2 - 10*np.cos(2*np.pi*vector))


def schwefel(vector:np.ndarray, dimensions:int=None):
    if dimensions is not None and dimensions != vector.shape[0]:
        raise Exception("Number of dimensions is different from the vector shape")
    d = vector.shape[0]
    
    return 418.9829*d - np.sum(vector*np.sin(np.sqrt(np.abs(vector))))        
        
def rosenbrock(vector: np.ndarray, dimensions: int = None):
    if dimensions is not None and dimensions != vector.shape[0]:
        raise Exception("Number of dimensions is different from the vector shape")

    x = vector[:-1]
    x_next = vector[1:]
    return np.sum(100 * (x_next - x**2)**2 + (x - 1)**2)
            
