import numpy as np 
import array  
import cnn_conjunctiveBRB_DE_combined_validation as dbrb   
import cnn_conjunctiveBRB_DE_combined_best as bestBRB
#len(bounds)         

f1 = open("sensor.txt", "r")
f2 = open("cnn.txt", "r")
f3 = open("aqi.txt", "r")
  
sensor = [10.35] * 150
cnn = [10.35] * 150
aqi = [10.35] * 150
#cnn = array.array('f', [10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, #10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32])

#aqi = array.array('f', [10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32, 10.32])
    #f = open("cnn_prediction2.txt", "r") #nominal 36 
    #f = open("cnn_prediction3.txt", "r") #mild 117
tk1 = 0
tk2 = 0
tk3 = 0   
 
if f1.mode == 'r':  
        #print("reading cnn_prediction.txt file \n")  
    f11 = f1.readlines()
    
    for line in f11:   
        sensor[tk1] = float(line)
        tk1 += 1
    #print(sensor[12])

else:
    print("Unable to open the file sensor.txt")
     
    
if f2.mode == 'r':
        #print("reading cnn_prediction.txt file \n")  
    f21 = f2.readlines() 
    
    for line in f21:   
        cnn[tk2] = float(line)
        tk2 += 1
    #print(cnn[11])     

else:
    print("Unable to open the file cnn.txt")
            

if f3.mode == 'r':
        #print("reading cnn_prediction.txt file \n")  
    f31 = f3.readlines()
    
    for line in f31:   
        aqi[tk3] = float(line)
        tk3 += 1
         
    #print(aqi[13])       

else:
    print("Unable to open the file aqi.txt")

fo = open("de_mse.txt", "w")
rec_test_aqi = open("predicted_aqi_mse.txt", "w")  
  
def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=50, its=300): #popsize = 170
    dimensions = 41   
    global best 
    pop = np.random.rand(popsize, dimensions)  
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff  
    #fitness = (70-np.asarray([fobj(ind) for ind in pop_denorm]))**2 #BRB_DE
    fitness = (np.asarray([fobj(ind) for ind in pop_denorm])) #BRB_DE 
    best_idx = np.argmin(fitness)  
    best = pop_denorm[best_idx] 
    for i in range(its): 
        for j in range(popsize):    
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1) 
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j]) 
            trial_denorm = min_b + trial * diff   
            #f = (70-fobj(trial_denorm))**2 #BRB_DE  
            f = (fobj(trial_denorm)) #BRB_DE 
            if f < fitness[j]:          
                fitness[j] = f 
                pop[j] = trial  
                if f < fitness[best_idx]:  
                    best_idx = j
                    best = trial_denorm
        if i == 99:
            fo.write("After 100 its, MSE is ")
            fo.write(str(fitness[best_idx])) 
            fo.write("\n")
        elif i == 199:
            fo.write("After 200 its, MSE is ")
            fo.write(str(fitness[best_idx]))  
            fo.write("\n")
        elif i == 299:
            fo.write("After 300 its, MSE is ")
            fo.write(str(fitness[best_idx])) 
            fo.write("\n")
        elif i == 399: 
            fo.write("After 400 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 499:
            fo.write("After 500 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 599:
            fo.write("After 600 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 699:
            fo.write("After 700 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 799:
            fo.write("After 800 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 899: 
            fo.write("After 900 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        elif i == 999: 
            fo.write("After 1000 its, MSE is ")
            fo.write(str(fitness[best_idx]))
            fo.write("\n")
        #yield best    
        yield best, fitness[best_idx]   
        #yield best, fitness[best_idx]
        #yield min_b + pop * diff, fitness, best_idx         
#it = list(de(lambda x: x**2, bounds=[(1, 20)]))                                  
def fobj(x):       
    sum = 0   
    for go in range(120):            
        pred_aqi = dbrb.ruleBase(sensor[go], cnn[go], x)
        sum += (aqi[go] - pred_aqi)**2    
    return sum/120           
    #dbrb.takeInput() 
    #aqi = dbrb.ruleBase(x)
    #t = fmodel(x)          
    #return (423-aqi)**2           

it = list(de(fobj, bounds=[(0, 1)]))   
print(it[-1])        
      
for g in range(41):    
    print("best[0-40] ", best[g])          
    
for parse in range(120, 150):
    test_MSE = 0
    test_pred_aqi = bestBRB.deRuleBase(sensor[parse], cnn[parse], best)
    test_MSE += (aqi[parse] - test_pred_aqi)**2      
    rec_test_aqi.write(str(test_pred_aqi))
    rec_test_aqi.write("\n")
    
mse_final = test_MSE/30
rec_test_aqi.write("MSE of Test Dataset under DE Conjunctive BRB is ")
rec_test_aqi.write(str(mse_final)) 