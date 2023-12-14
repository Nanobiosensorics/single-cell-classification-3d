import time

def log(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        print('Generated', args[0], round(end-start, 2), 's')
        return result
    return wrap