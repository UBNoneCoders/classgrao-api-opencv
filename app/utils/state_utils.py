running = False

def get_running():
    return running

def set_running(valor: bool):
    global running
    running = valor
