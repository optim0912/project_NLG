import psutil

def memory_usage():
    # current process RAM usage
    p = psutil.Process()
    rss = p.memory_info().rss / 2 ** 20 # Bytes to MB
    print(f"current memory usage: {rss: 10.5f} MB")