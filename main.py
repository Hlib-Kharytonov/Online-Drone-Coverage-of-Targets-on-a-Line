import Environment
import matplotlib.pyplot as plt
import math


if __name__ == "__main__":
    print("=== test simulation ===")
    field = Environment.Environment()
    field.spawn_requests(nb_requests=1000, mu=100.0, sig=100.0)
    
    print(f"{len(field.requests)} requests has been generated. Their coordinatest:") # test the order of points
    for i, req in enumerate(field.requests):
       print(f"request {i+1}: x = {req.x:.2f}")
       
       
    print("=== STRAIGHT-UP algorithm test ===")
    for i, req in enumerate(field.requests):
        # print(f"iteration {i}:")
        field.drone.straight_up_algorithm(req.x)
        
    n=len(field.requests)
    plt.plot([r.x for r in field.requests],[0]*n,"ko",label="Targets")
    x,y = zip(*field.drone.movement_track)
    
    dist_straight = field.drone.total_distance
    
    plt.plot(x,y,"r-", label="STRAIGHT-UP")
    # plt.show()
        
        
    field.drone.reset()
    print("=== GREEDY algorithm test ===")
    for i, req in enumerate(field.requests):
        # print(f"iteration {i}:")
        field.drone.greedy_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    dist_greedy = field.drone.total_distance
    
    plt.plot(x,y,"b-", label="GREEDY")
    
    field.drone.reset()
    print("=== BETA-HEDGE algorithm test ===")
    for i, req in enumerate(field.requests):
        # print(f"iteration {i}:")
        field.drone.beta_hedge_algorithm(req.x)


    x,y = zip(*field.drone.movement_track)
    
    dist_beta = field.drone.total_distance
    
    plt.plot(x,y,"g-", label="BETA-HEDGE")
    
    
    field.drone.reset()
    print("=== MEAN-HEDGE algorithm test ===")
    for i, req in enumerate(field.requests):
        # print(f"iteration {i}:")
        field.drone.mean_hedge_algo(req.x)


    x,y = zip(*field.drone.movement_track)
    
    dist_median = field.drone.total_distance
    
    plt.plot(x,y,"y-", label="MEAN-HEDGE")
    
     
    field.drone.reset()
    print("=== LEARNING BETA UP algorithm test ===")
    for i, req in enumerate(field.requests):
        # print(f"iteration {i}:")
        field.drone.learning_beta_up_algorithm(req.x)
        
    x,y = zip(*field.drone.movement_track)
    
    dist_LBA = field.drone.total_distance
    
    plt.plot(x,y,"ro-", label="LEARNING STRAIGHT UP")
    
    
    
    field.drone.reset()
    print("=== LEARNING GREEDY UP algorithm test ===")
    for i, req in enumerate(field.requests):
        # print(f"iteration {i}:")
        field.drone.learning_greedy_up_algorithm(req.x)
        
    x,y = zip(*field.drone.movement_track)
    
    dist_LGA = field.drone.total_distance
    
    plt.plot(x,y,"bo-", label="LEARNING STRAIGHT UP")
    
    print("\n" + "="*30)
    print("results::")
    print(f"STRAIGHT-UP:  {dist_straight:.2f}")
    print(f"GREEDY:       {dist_greedy:.2f}")
    print(f"BETA-HEDGE:   {dist_beta:.2f}")
    print(f"MEAN-HEDGE: {dist_median:.2f}")
    print(f"LEARNING BETA UP: {dist_LBA:.2f}")
    print(f"LEARNING GREEDY UP: {dist_LGA:.2f}")
    print("="*30 + "\n")
    
    
    final_x = field.drone.x
    final_y = field.drone.y

    coverage_radius = final_y * math.tan(field.drone.alpha)

    plt.plot([final_x, final_x - coverage_radius], [final_y, 0], "g--", alpha=0.5)
    plt.plot([final_x, final_x + coverage_radius], [final_y, 0], "g--", alpha=0.5)
    
    
    plt.legend(loc="best")
    plt.show()
    # plt.savefig('my_simulation.pdf')
    
    
