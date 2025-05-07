import datetime

def nowstr():
    """Return the current date and time as a string."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")