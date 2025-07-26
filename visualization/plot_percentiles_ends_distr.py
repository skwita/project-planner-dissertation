import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from services.metrics import monte_carlo_simulation

def plot_percentile_distributions(durations, percentile, filename):
    plt.figure(figsize=(12, 8))
    
    plt.hist(durations, bins=1000, range=(0, 200))
    
    plt.title('Распределение длительности проекта по процентилям')
    plt.xlabel('Длительность проекта (дни)')
    plt.ylabel('Плотность вероятности')
    plt.title(f'Процентиль: {percentile}')
    plt.grid(True)
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig(filename)
    plt.close()