import Environment
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

if __name__ == "__main__":
    random.seed(17)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 7))
    print("=== test simulation ===")
    field = Environment.Environment()
    field.spawn_requests(nb_requests=500, mu=100.0, sig=100.0)
    
    print(f"{len(field.requests)} requests has been generated. Their coordinatest:") # test the order of points
    for i, req in enumerate(field.requests):
       print(f"request {i+1}: x = {req.x:.2f}")
       
    print("=== STRAIGHT-UP algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.straight_up_algorithm(req.x)
        
    n=len(field.requests)
    ax3.plot([r.x for r in field.requests],[0]*n,"ko",label="Targets")
    x,y = zip(*field.drone.movement_track)
    
    dist_straight = field.drone.total_distance
    
    ax3.plot(x,y,"r-", label="STRAIGHT-UP")

        
        
    field.drone.reset()
    print("=== GREEDY algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.greedy_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    dist_greedy = field.drone.total_distance
    
    ax3.plot(x,y,"b-", label="GREEDY")
    
    field.drone.reset()
    print("=== BETA-HEDGE algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.beta_hedge_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    dist_beta = field.drone.total_distance
    
    ax3.plot(x,y,"g-", label="BETA-HEDGE")
    
    
    field.drone.reset()
    print("=== MEAN-HEDGE algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.mean_hedge_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    dist_median = field.drone.total_distance
    
    ax3.plot(x,y,"y-", label="MEAN-HEDGE")
    
     
    field.drone.reset()
    print("=== LEARNING BETA UP algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.learning_beta_up_algorithm(req.x)
        
    x,y = zip(*field.drone.movement_track)
    
    dist_LBA = field.drone.total_distance
    
    ax3.plot(x,y,"ro-", label="LEARNING STRAIGHT UP")
    
    
    
    field.drone.reset()
    print("=== LEARNING GREEDY UP algorithm test ===")
    for i, req in enumerate(field.requests):
        print(f"iteration {i}:")
        field.drone.learning_greedy_up_algorithm(req.x)
        
    x,y = zip(*field.drone.movement_track)
    
    dist_LGA = field.drone.total_distance
    
    ax3.plot(x,y,"bo-", label="LEARNING GREEDY UP")
    # 
    # print("\n" + "="*30)
    # print("results::")
    # print(f"STRAIGHT-UP:  {dist_straight:.2f}")
    # print(f"GREEDY:       {dist_greedy:.2f}")
    # print(f"BETA-HEDGE:   {dist_beta:.2f}")
    # print(f"MEAN-HEDGE: {dist_median:.2f}")
    # print(f"LEARNING BETA UP: {dist_LBA:.2f}")
    # print(f"LEARNING GREEDY UP: {dist_LGA:.2f}")
    # print("="*30 + "\n")
    
    
    final_x = field.drone.x
    final_y = field.drone.y

    coverage_radius = final_y * math.tan(field.drone.alpha)

    ax3.plot([final_x, final_x - coverage_radius], [final_y, 0], "g--", alpha=0.5)
    ax3.plot([final_x, final_x + coverage_radius], [final_y, 0], "g--", alpha=0.5)
    ax3.set_title(r'trajectories of the drone for different algorithms')
    ax3.legend(loc=2, fontsize='small')


#======================================== Mu dependence test =========================================

    stats = {
        "STRAIGHT-UP": [],
        "GREEDY": [],
        "BETA-HEDGE": [],
        "MEAN-HEDGE": [],
        "LEARNING BETA UP": [],
        "LEARNING GREEDY UP": [] }
    
    mu_samples = [0, 10, 100, 500, 1000]
    for mu in mu_samples:
        field = Environment.Environment()
        field.spawn_requests(nb_requests=500, mu=mu, sig=100.0)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.straight_up_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["STRAIGHT-UP"].append(tmp/10)
        
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.greedy_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["GREEDY"].append(tmp/10)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.beta_hedge_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["BETA-HEDGE"].append(tmp/10)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.mean_hedge_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["MEAN-HEDGE"].append(tmp/10)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.learning_beta_up_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["LEARNING BETA UP"].append(tmp/10)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.learning_greedy_up_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["LEARNING GREEDY UP"].append(tmp/10)

    
    df = pd.DataFrame(stats, index=mu_samples)
    ax1.plot(mu_samples, df, marker='o')
    ax1.legend(df.columns, loc=2, fontsize='small')
    ax1.set_title(r'dependence of mu on distance flown, sigma=100')
    ax1.set_xlabel(r'Esperance($\mu$)')
    ax1.set_ylabel('Distance Flown')

    fig.suptitle(r'Effectiveness of Algorithms', fontsize=14, y=0.95)


#======================================== SIGMA dependence test =========================================
    stats = {
        "STRAIGHT-UP": [],
        "GREEDY": [],
        "BETA-HEDGE": [],
        "MEAN-HEDGE": [],
        "LEARNING BETA UP": [],
        "LEARNING GREEDY UP": [] }
    
    sigma_samples = [10, 50, 100, 150, 500]
    for sigma in sigma_samples:
        field = Environment.Environment()
        field.spawn_requests(nb_requests=500, mu=0, sig=sigma)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.straight_up_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["STRAIGHT-UP"].append(tmp/10)
        
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.greedy_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["GREEDY"].append(tmp/10)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.beta_hedge_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["BETA-HEDGE"].append(tmp/10)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.mean_hedge_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["MEAN-HEDGE"].append(tmp/10)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.learning_beta_up_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["LEARNING BETA UP"].append(tmp/10)
        
        tmp = 0
        for i in range(10):
            for req in field.requests:
                field.drone.learning_greedy_up_algorithm(req.x)
            tmp = tmp + field.drone.total_distance
            field.drone.reset()
        stats["LEARNING GREEDY UP"].append(tmp/10)

    
    df = pd.DataFrame(stats, index=sigma_samples)
    ax2.plot(sigma_samples, df, marker='o')
    ax2.legend(df.columns, loc=2, fontsize='small')
    ax2.set_title(r'dependence of sigma on distance flown, mu=0')
    ax2.set_xlabel(r'Esperance($\sigma$)')
    ax2.set_ylabel('Distance Flown')

    # plt.show()

    plt.savefig('my_simulation.pdf')
    
    
