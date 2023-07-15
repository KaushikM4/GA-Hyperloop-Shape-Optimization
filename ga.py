
import os
import subprocess
import numpy as np
import pandas as pd
import math
import scipy
from scipy.special import factorial
import matplotlib.pyplot as plt
import random
import subprocess

# Compile the C++ code
subprocess.run(["g++", "pressure_drag.cpp", "-o", "pressure_drag_py"])

#Point to be varied is P1
n_iter = 30    #number of generations to run
pop_size=100    #population size
x_low_lim=0.5   #lower limit of x coordinate of chosen control point
x_up_lim=1.5    #upper limit of x 
y_low_lim=0.5   #lower limit of y
y_up_lim=1.5    #upper limit of y 
str_len=16      #length of encoded bit string
num_points=80   #number of points on Bezier curve(should match the number in c++ code)
k=10             #k-way tournament selection
mut_prob=0.3    #mutation probability
x_width=x_up_lim-x_low_lim
y_width=y_up_lim-y_low_lim
def drag_compute(file_name):
    # Run the compiled code and capture its output
    result = subprocess.run("./pressure_drag_py", capture_output=True, text=True)
    # Print the output of the C++ code
    drag=result.stdout
    #print(drag+"\n")
    return drag


#Just for Visualisation
class Control:
    def __init__(self,x,y):
        self.x=x
        self.y=y
P0=Control(0.0,0.0)
P1=Control(1.0,2.0)
P2=Control(1.0,4.0)
P3=Control(0.0,6.0)

t=np.linspace(0,1,num_points)
def Bezier(P0, P1, P2, P3, t):
    
    t1 = P0.x*( - pow(t, 3) + 3*pow(t, 2) - 3*t + 1)
    t2 = P1.x * (3*pow(t, 3) - 6 * pow(t, 2) + 3 * t)
    t3 = P2.x * (-3*pow(t, 3) + 3 * pow(t, 2))
    t4 = P3.x * (pow(t, 3))

    P_x = t1+t2+t3+t4
    
    t1 = P0.y * (-pow(t, 3) + 3 * pow(t, 2) - 3 * t + 1)
    t2 = P1.y * (3 * pow(t, 3) - 6 * pow(t, 2) + 3 * t)
    t3 = P2.y * (-3 * pow(t, 3) + 3 * pow(t, 2))
    t4 = P3.y * (pow(t, 3))

    P_y = t1 + t2 + t3 + t4

    return P_x,P_y

#print(Bezier(P0, P1, P2, P3, 1))
Bezier_curve=np.vectorize(Bezier)
#curve_x,curve_y=Bezier_curve(P0, P1, P2, P3, t)
#plt.plot(curve_x,curve_y)

def population(P1_x,P1_y):
    drag=np.zeros(pop_size)
    coord=np.zeros((pop_size,num_points,2))
    min=0
    for i in range(pop_size):
        P1.x=P1_x[i]
        P1.y=P1_y[i]
        print(P1.x,P1.y)
        indiv=i+1
        print("gen"+str(gen)+"indiv"+str(indiv))
        coord[i,:,:]=np.transpose(Bezier_curve(P0, P1, P2, P3, t))
        # a=plt.figure()
        # a.figwidth=2.5
        # a.figheight=2.5
        # plt.plot(-coord[i,:,0],-coord[i,:,1])
        #plt.show()
        filename="P2.txt"
        Control_file=open(filename,'w')
        Control_file.write(str(P1.x)+"\n"+str(P1.y))
        Control_file.close()
        drag[i]=drag_compute(filename)
        if drag[i]<drag[min]:
            min=i
    return (drag,min,coord)
#(drag,min,coord)=population(P1_x,P1_y)
#print(f'Minimum drag of gen{gen} obtained at P2 coordinates ({P1_x[min]},{P1_y[min]})\nDrag={drag[min]}kN')
#plt.plot(-coord[min,:,0],-coord[min,:,1])



#Sorting individuals based on drag value.The ones offering lowest drag are brought to the beginning
def sort_pod(P1_x,P1_y,drag):
    indices = np.argsort(drag)
    P1_xs=P1_x[indices]
    P1_ys=P1_y[indices]
    drag_s=drag[indices]
    #print(indices)
    #for i in range(num_points):
        #print(f'({P1_xs[i]},{P1_ys[i]}),drag={drag_s[i]}kN')
    return P1_xs,P1_ys
#P1_xs,P1_ys=sort_pod(P1_x,P1_y,drag)

#Binary Encoding
def bin_enc(P1_xs,P1_ys,str_len):
    #Scaling to an 8-bit binary string
    P1_xsb=(P1_xs-x_low_lim)*(2**str_len-1)/x_width
    P1_ysb=(P1_ys-y_low_lim)*(2**str_len-1)/y_width
    P1_xsb=P1_xsb.astype('int32')
    P1_ysb=P1_ysb.astype('int32')
    # print(np.amax(P1_xsb),np.amin(P1_xsb))
    # print(np.amax(P1_ysb),np.amin(P1_ysb))
    binary_repr_v = np.vectorize(np.binary_repr)
    P1_xsb=binary_repr_v(P1_xsb,str_len)
    P1_ysb=binary_repr_v(P1_ysb,str_len)
    # print(len(P1_xsb),len(P1_ysb))
    # print(P1_xsb,P1_ysb)
    return (P1_xsb,P1_ysb)
# (P1_xsb,P1_ysb)=bin_enc(P1_xs,P1_ys,str_len)
# print(type(P1_xsb[0]))

#k way Tournament Selection
def tournament_selection(k):
    parents=np.random.randint(0,pop_size, size=(int(pop_size/2),k))
    parents = np.sort(parents,axis=1)
    parents=parents[:,[0,1]]
    return parents
#ind_new_gen=tournament_selection(5)

#Crossover
#crossover probability of 100 percent
def crossover(P1_xsb,P1_ysb,ind_new_gen,cross_p=1):
    P1_xsbc=P1_xsb
    P1_ysbc=P1_ysb
    #print("before",P1_xsbc[50])
    for i in range(1,int(pop_size/2)):#preserve the best 2
        #print("before",WLsbn[2*i])
        cross_p=random.randint(0,len(P1_xsb[0]))
        #cross_p=10
        parx1=(P1_xsb[ind_new_gen[i,0]])
        parx2=(P1_xsb[ind_new_gen[i,1]])
        P1_xsbc[2*i]=parx1[0:cross_p]+parx2[cross_p:]
        P1_xsbc[2*i+1]=parx2[0:cross_p]+parx1[cross_p:]
        pary1=(P1_ysb[ind_new_gen[i,0]])
        pary2=(P1_ysb[ind_new_gen[i,1]])
        P1_ysbc[2*i]=pary1[:cross_p]+pary2[cross_p:]
        P1_ysbc[2*i+1]=pary2[:cross_p]+pary1[cross_p:]
        #print("after",WLsbn[2*1])
    return (P1_xsbc, P1_ysbc)
#P1_xsbc, P1_ysbc=crossover(P1_xsb,P1_ysb,ind_new_gen,1)
#print("after",P1_xsbc[50])

#Mutation
def mutation(P1_xsbc, P1_ysbc,mut_prob):
    mut_child=np.random.randint(2,pop_size,size=int(pop_size*mut_prob))#preserve the best 2
    #print(mut_child)
    P1_xsbcm=P1_xsbc
    P1_ysbcm=P1_ysbc
    for i in range(0,int(pop_size*mut_prob)):
        mut_p=random.randint(0,len(P1_xsbc[0])-1)
        #print(mut_p)
        str1=P1_xsbc[mut_child[i]]
        #print(len(str1))
        str2=P1_ysbc[mut_child[i]]
        #print("before",P1_xsbcm[mut_child[i]])
        if str1[mut_p]=='0':
            str1=str1[:mut_p]+'1'+str1[mut_p+1:]
        else:
            str1=str1[:mut_p]+'0'+str1[mut_p+1:]
        if str2[mut_p]=='0':
            str2=str2[:mut_p]+'1'+str1[mut_p+1:]
        else:
            str2=str2[:mut_p]+'0'+str1[mut_p+1:]
        P1_xsbcm[mut_child[i]]=str1
        P1_ysbcm[mut_child[i]]=str2
        #print("after",P1_xsbcm[mut_child[i]])
    return (P1_xsbcm, P1_ysbcm)
#P1_xsbcm, P1_ysbcm=mutation(P1_xsbc, P1_ysbc,0.2)
        

#Decoding back to coordinates
def decode(P1_xsbcm, P1_ysbcm):
    def convert(str1,str2):
        ex=int(str1,2)
        ey=int(str2,2)
        return (ex,ey)
    decode=np.vectorize(convert)
    (P1_xn, P1_yn)=decode(P1_xsbcm, P1_ysbcm)
    P1_xn=(P1_xn*x_width)/(2**str_len-1)+x_low_lim
    P1_yn=(P1_yn*y_width)/(2**str_len-1)+y_low_lim
    return (P1_xn, P1_yn)
#(P1_xn, P1_yn)=decode(P1_xsbcm, P1_ysbcm)
#print(P1_xn, P1_yn)

#Convergence Condition/Loop
P1_x=x_low_lim+(np.random.rand(pop_size))*x_width
P1_y=y_low_lim+(np.random.rand(pop_size))*y_width
min_drag=np.zeros((n_iter,3))
for gen in range(1,n_iter+1):
    print("gen+"+str(gen))
    (drag,min,coord)=population(P1_x,P1_y)
    min_drag[gen-1,0]=P1_x[min]
    min_drag[gen-1,1]=P1_y[min]
    min_drag[gen-1,2]=drag[min]
    if gen==n_iter:
        break
    (P1_xs,P1_ys)=sort_pod(P1_x,P1_y,drag)
    (P1_xsb,P1_ysb)=bin_enc(P1_xs,P1_ys,str_len)
    ind_new_gen=tournament_selection(k)
    (P1_xsbc, P1_ysbc)=crossover(P1_xsb,P1_ysb,ind_new_gen,1)
    (P1_xsbcm, P1_ysbcm)=mutation(P1_xsbc, P1_ysbc,mut_prob)
    (P1_xn, P1_yn)=decode(P1_xsbcm, P1_ysbcm)
    (P1_x,P1_y)=(P1_xn, P1_yn)
print(min_drag)
min=np.argmin(min_drag[:,2])
print(f'Convergence obtained at gen{min+1} obtained at P2 coordinates ({min_drag[min,0]},{min_drag[min,1]})\nDrag={min_drag[min,2]}kN')

#saving drag(fitness function) values excel file for plotting
df = pd.DataFrame (min_drag[:,2])
filepath = 'C:/Users\91944/Downloads/GA Shape Optimization/Opti Project/fitness_2.xlsx'
df.to_excel(filepath, index=False)
