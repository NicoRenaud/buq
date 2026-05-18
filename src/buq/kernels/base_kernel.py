
class BaseKernel:
    def __init__(self):
        pass
    
    def get_kernel(self, x_data, y_data, bounds):
        raise NotImplementedError
