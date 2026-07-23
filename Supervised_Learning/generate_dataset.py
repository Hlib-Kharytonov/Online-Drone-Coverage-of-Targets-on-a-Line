import numpy as np
import pandas as pd
import math
from Environment import Environment 

def generate_data():
    
    n_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    mu_values = [0, 10, 20, 30, 50, 100, 150, 200]
    sigma_values = [10, 20, 30, 50, 100, 150, 200]
    
    env = Environment()
    alpha = env.AOV
    
    possible_betas = np.linspace(0, alpha * 0.95, 15) 
    
    dataset = []
    total_iterations = len(n_values) * len(mu_values) * len(sigma_values)
    current_iter = 0

    print(f"Starting dataset generation ({total_iterations} combinations). This will take some time...")

    # 2. Перебираем все возможные комбинации
    for n in n_values:
        for mu in mu_values:
            for sigma in sigma_values:
                current_iter += 1
                best_betas_for_this_setup = []
                
                # Для надежности прогоняем каждую комбинацию 5 раз
                for _ in range(5):
                    # Генерируем новый кластер точек
                    env.spawn_requests(n, mu, sigma)
                    
                    best_beta = 0
                    min_distance = float('inf')
                    
                    
                    for test_beta in possible_betas:
                        env.drone.reset()
                        
                        
                        for req in env.requests:
                            env.drone.learning_beta_up_algorithm(req.x, custom_beta=test_beta)
                        
                        flown_distance = env.drone.total_distance
                        
                        if flown_distance < min_distance:
                            min_distance = flown_distance
                            best_beta = test_beta
                            
                    best_betas_for_this_setup.append(best_beta)
                
                average_best_beta = sum(best_betas_for_this_setup) / len(best_betas_for_this_setup)
                
                dataset.append({
                    'n': n,
                    'mu': mu,
                    'sigma': sigma,
                    'best_beta': average_best_beta
                })
                
                if current_iter % 50 == 0:
                    print(f"Progress: {current_iter} / {total_iterations}...")

    # 5. Сохраняем в CSV
    df = pd.DataFrame(dataset)
    df.to_csv("drone_dataset.csv", index=False)
    print("Done! Dataset saved to 'drone_dataset.csv'.")

if __name__ == "__main__":
    generate_data()