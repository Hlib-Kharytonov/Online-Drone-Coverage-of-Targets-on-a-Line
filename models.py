import math
import statistics
import numpy as np

class Request:
    
    def __init__(self,x, feas_cone):
        self.x= x
        self.feas_cone=feas_cone 

class Drone:
    def __init__(self):
        self.x = self.y = 0.0
        
        self.alpha = math.pi/4          #you can change your AOV here, but you need qs well to change corresponding beta ange just below
        self.beta = math.pi/9.666		#as our drone has fixed  AOV of pi/4 
        
        self.total_distance = 0.0
        self.min_x_seen = 0.0
        self.max_x_seen = 0.0
        self.movement_track=[(0,0)]
        self.drone_memory = []
        self.angles = np.linspace(0, self.alpha * 0.90, 10)
        self.weights = [1.0] * len(self.angles)
        self.eta = 0.1

        
    def reset(self):
        self.x = self.y = 0.0
        self.total_distance = 0.0
        self.min_x_seen = 0.0
        self.max_x_seen = 0.0
        self.movement_track=[(0,0)]
        self.drone_memory = []
        self.weights = [1.0] * len(self.angles)
        self.beta = math.pi/9.666
        self.alpha = math.pi/4
        
    def get_coverage_radius(self):
        return self.y * math.tan(self.alpha)


    def move_straight_up(self, target_y ):
        
        if target_y > self.y:
            distance_flown = target_y - self.y
            self.total_distance += distance_flown
            self.y = target_y
            self.movement_track.append((0,self.y))
            print(f"Drone moved to ({self.x:.2f},{self.y:.2f}), total distance: {self.total_distance:.2f}")
            
        else:
            print("drone didn't move")


    def straight_up_algorithm(self, target_x):
        horizontalDist = abs(self.x - target_x)
        requiredY = horizontalDist / math.tan(self.alpha)
        self.move_straight_up(requiredY) 


    def move_zigzag(self, target_x, target_y):
        
        if target_y >= self.y:
            distance_flown = math.dist((self.x, self.y),(target_x,target_y))
            self.total_distance +=distance_flown
            self.x = target_x
            self.y = target_y
            self.movement_track.append((self.x,self.y))
            print(f"Drone moved to ({self.x:.2f},{self.y:.2f}). total distance: {self.total_distance:.2f}")
        
        else:
            print("drone didn't moove")
    
    def greedy_algorithm(self,target_x):
        self.min_x_seen = min(self.min_x_seen, target_x)
        self.max_x_seen = max(self.max_x_seen, target_x)
        
        apex_x = (self.min_x_seen + self.max_x_seen)/2.0
        
        distance_to_edge = (self.max_x_seen - self.min_x_seen)/2.0
        requiredY = distance_to_edge / math.tan(self.alpha)

        self.move_zigzag(apex_x, requiredY)
        
    def calculate_dist_beta(self,r):
        return (math.tan(self.alpha)+(1+2*r)*math.tan(self.beta)) / (((math.tan(self.alpha)+math.tan(self.beta))**2)*math.cos(self.beta))
        
        
    def beta_hedge_algorithm(self, target_x):

        self.beta = math.pi/9.666
        
        L = self.min_x_seen = min(self.min_x_seen,target_x)
        R = self.max_x_seen = max(self.max_x_seen,target_x)
        
        current_coverage = self.get_coverage_radius()
        
        if self.x - current_coverage <= self.min_x_seen and self.x + current_coverage >= self.max_x_seen:
            return

        apex_x = (self.min_x_seen + self.max_x_seen) / 2.0
        direction = 1 if apex_x > self.x else -1
        
        c = math.tan(self.alpha)
        k= math.tan(self.beta)
        
        if direction == 1:
            y_left = (self.x - L - self.y * k) / (c - k)
            y_right = (R - self.x + self.y * k) / (c + k)
        else:
            y_left = (self.x - L + self.y * k) / (c + k)
            y_right = (R - self.x - self.y * k) / (c - k)
        
        target_y = max(max(y_left, y_right), self.y)

        target_x_final = self.x + direction * (target_y - self.y) * k

        self.move_zigzag(target_x_final, target_y)
        
        
    def mean_hedge_algorithm(self, target_x):
        
        self.beta = math.pi/9.666

        self.drone_memory.append(target_x)
        L = self.min_x_seen = min(self.min_x_seen,target_x)
        R = self.max_x_seen = max(self.max_x_seen,target_x)
        
        current_coverage = self.get_coverage_radius()
        
        if self.x - current_coverage <= self.min_x_seen and self.x + current_coverage >= self.max_x_seen:
            return
        
        c = math.tan(self.alpha)
        k= math.tan(self.beta)
        
        mean = statistics.mean(self.drone_memory)
        direction = 1 if mean >= self.x else -1
     
        if direction == 1:
            y_left = (self.x - L - self.y * k) / (c - k)
            y_right = (R - self.x + self.y * k) / (c + k)
        else:
            y_left = (self.x - L + self.y * k) / (c + k)
            y_right = (R - self.x - self.y * k) / (c - k)
            
        target_y = max(max(y_left, y_right), self.y)

        target_x_final = self.x + direction * (target_y - self.y) * k
        self.move_zigzag(target_x_final, target_y)
        
    def learning_beta_up_algorithm(self, target_x):

        self.beta = math.pi/9.666
        
        self.drone_memory.append(target_x)
        L = self.min_x_seen = min(self.min_x_seen,target_x)
        R = self.max_x_seen = max(self.max_x_seen,target_x)
        
        current_coverage = self.get_coverage_radius()
        
        if self.x - current_coverage <= self.min_x_seen and self.x + current_coverage >= self.max_x_seen:
            return
        
        c= math.tan(self.alpha)
        k= math.tan(self.beta)
        
         
        mean = statistics.mean(self.drone_memory)
        direction = 1 if mean >= self.x else -1
        
        if direction == 1:
            y_left = (self.x - L - self.y * k) / (c - k)
            y_right = (R - self.x + self.y * k) / (c + k)
        else:
            y_left = (self.x - L + self.y * k) / (c + k)
            y_right = (R - self.x - self.y * k) / (c - k)
            
        target_y = max(max(y_left, y_right), self.y)

        x_beta = self.x + direction * (target_y - self.y) * k
        
        
        x = mean
        
        if (self.x <= mean and mean <= x_beta):
            y = (R - x)/c
            self.move_zigzag(x,y)
        elif (x_beta <= mean and mean <= self.x):
            y = (x-L)/c
            self.move_zigzag(x,y)
        else:
            self.beta_hedge_algorithm(target_x)
        return
        
        
        
    def learning_greedy_up_algorithm(self, target_x):
        
        self.drone_memory.append(target_x)
        L = self.min_x_seen = min(self.min_x_seen,target_x)
        R = self.max_x_seen = max(self.max_x_seen,target_x)
        
        current_coverage = self.get_coverage_radius()
        
        if self.x - current_coverage <= self.min_x_seen and self.x + current_coverage >= self.max_x_seen:
            return
        
        c= math.tan(self.alpha)
        k= math.tan(self.beta)
        
         
        mean = statistics.mean(self.drone_memory)
        
        apex_x = (self.min_x_seen + self.max_x_seen) / 2.0
        
        
        x = mean
        
        if (self.x <= mean and mean <= apex_x):
            y = (R - x)/c
            self.move_zigzag(x,y)
        elif (apex_x <= mean and mean <= self.x):
            y = (x-L)/c
            self.move_zigzag(x,y)
        else:
            self.greedy_algorithm(target_x)
        return
        

    def MWU_algorithm(self, target_x):
        self.drone_memory.append(target_x)
        L = self.min_x_seen = min(self.min_x_seen, target_x)
        R = self.max_x_seen = max(self.max_x_seen, target_x)
        
        current_coverage = self.get_coverage_radius()
        
        if self.x - current_coverage <= self.min_x_seen and self.x + current_coverage >= self.max_x_seen:
            return
        
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        
        chosen_idx = np.random.choice(len(self.angles), p=probabilities)
        self.beta = self.angles[chosen_idx]

        apex_x = (self.min_x_seen + self.max_x_seen) / 2.0
        direction = 1 if apex_x > self.x else -1
        c = math.tan(self.alpha)

        costs = []
        
        chosen_target_x = 0
        chosen_target_y = 0

        
        for i, angle in enumerate(self.angles):
            k = math.tan(angle)
            if direction == 1:
                y_left = (self.x - L - self.y * k) / (c - k)
                y_right = (R - self.x + self.y * k) / (c + k)
            else:
                y_left = (self.x - L + self.y * k) / (c + k)
                y_right = (R - self.x - self.y * k) / (c - k)
            
            target_y = max(max(y_left, y_right), self.y)
            target_x_final = self.x + direction * (target_y - self.y) * k

            cost = math.dist((self.x, self.y), (target_x_final, target_y))
            costs.append(cost)

            if i == chosen_idx:
                chosen_target_x = target_x_final
                chosen_target_y = target_y

        max_cost = max(costs)
        
        if max_cost > 0:
            for i in range(len(self.weights)):
                normalized_cost = costs[i] / max_cost 
                self.weights[i] = self.weights[i] * math.exp(-self.eta * normalized_cost)


        self.move_zigzag(chosen_target_x, chosen_target_y)










        
        