#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:45:25 2019

PDE: -u''=f in (0,1)
      DIrichlet boundary condition

Use linear Lagrange Finite Element Method to solve it.

Note: There might be a more efficient way to write the code. One possibility is to 
      use sparse matrix for the system matrix. You may also use iterative method to 
      solve the linear system. 
      You can also calculate its convergence rate.

@author: jiuhuahu
"""
import numpy as np
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt

class Quad_1D:
    def __init__(self,num_points):
        self.nq=num_points
    def getQuadOnRef_1D(self):
        if self.nq==1:
            self.what=np.array([1.0])
            self.xhat=np.array([0.5])
        elif self.nq==2:
            self.what=np.array([0.5, 0.5])
            self.xhat=np.array([0.211324865405187117745425609749021, 
                       0.788675134594812882254574390250978])
        elif self.nq==3:
            self.what=np.array([0.444444444444444444444444444444444,
                      0.277777777777777777777777777777777,
                      0.277777777777777777777777777777777])
            self.xhat=np.array(
                    [0.500000000000000000000000000000000,
                      0.112701665379258311482073460021760,
                      0.887298334620741688517926539978239])
        elif self.nq==4:
            self.what=np.array(
                    [0.326072577431273071313468025389000,
                     0.326072577431273071313468025389000,
                     0.173927422568726928686531974610999,
                     0.173927422568726928686531974610999])
            self.xhat=np.array(
                    [0.330009478207571867598667120448377,
                     0.669990521792428132401332879551622,
                     0.069431844202973712388026755553595,
                     0.930568155797026287611973244446404])
        elif self.nq==5:
            self.what=np.array(
                    [0.284444444444444444444444444444444,
                     0.239314335249683234020645757417819,
                     0.239314335249683234020645757417819,
                     0.118463442528094543757132020359958,
                     0.118463442528094543757132020359958])
            self.xhat=np.array(
                    [0.500000000000000000000000000000000,
                     0.230765344947158454481842789649895,
                     0.769234655052841545518157210350104,
                     0.046910077030668003601186560850303,
                     0.953089922969331996398813439149696])
        elif self.nq==6:
            self.what=np.array(
                    [0.180380786524069303784916756918858,
                     0.180380786524069303784916756918858,
                     0.233956967286345523694935171994775,
                     0.233956967286345523694935171994775,
                     0.085662246189585172520148071086366,
                     0.085662246189585172520148071086366])
            self.xhat=np.array(
                    [0.830604693233132256830699797509952,
                     0.169395306766867743169300202490047,
                     0.380690406958401545684749139159644,
                     0.619309593041598454315250860840355,
                     0.033765242898423986093849222753002,
                     0.966234757101576013906150777246997])
        else:
            print("Error will occur! Keep finding quadrature rule with more quadrature points")
            self.what=0
            self.xhat=0
                
            
class Dof_handler:
    
    def __init__(self,n,a,b):
        self.n=int(n) # n: number of cells
        self.a=a
        self.b=b
        
    def get_dof_inf(self):
         self.n_dof=int(self.n)+1
         self.n_elements=int(self.n)
         self.coordinates=np.zeros(self.n_dof)       
         self.coordinates=np.linspace(self.a,self.b,self.n_dof)
         self.dof=np.arange(self.n_dof)
         self.dof_bnd=[0,self.n_dof-1]
         self.dof_int=np.setdiff1d(self.dof,self.dof_bnd)
         self.elements=np.zeros((self.n_dof-1, 2))
         self.elements[:,0]=np.arange(self.n_dof-1)
         self.elements[:,1]=np.arange(self.n_dof-1)+1
         self.elements=self.elements.astype(int)


def rhs_func(x):
    #return 1.0/2.0*math.sin(2*math.pi*x )
    #return -math.exp(x)*(math.sin(math.pi*x)*(1-math.pi**2)+2*math.pi*math.cos(x*math.pi))#test example 1
    #return 2 #test example 2
    #return 6*x #test example 3
    return -6*x #test example 4



def exact_solution(x):
    #return math.exp(x)*math.sin(math.pi*x) #test example 1
    #return x-x**2 #test example 2
    #return x-x**3 #test example 3
    return x+x**3 #test example 4


##################################################################################3

Quad=Quad_1D(3)
Quad.getQuadOnRef_1D()


a=0
b=1
n=100    
p=1 # linear Lagrange polynomial
        
DOF=Dof_handler(n,a,b) 
DOF.get_dof_inf()

A=np.zeros((DOF.n_dof,DOF.n_dof))
F=np.zeros(DOF.n_dof)
uh=np.zeros(DOF.n_dof)
u=np.zeros(DOF.n_dof)

for cell in range(DOF.n_elements):
    dofIndices=DOF.elements[cell,:]
    dof_coordinates=DOF.coordinates[dofIndices]
    
    
    #-----assemble local stiffness matrix
    
    cell_matrix=np.zeros((p+1,p+1))
    for q in range(Quad.nq):               
        grad_ref_at_quad=np.array([-1, 1 ])   
        grad_ref_at_quad=grad_ref_at_quad.reshape(2,1)
        cell_matrix+=Quad.what[q]*1/(dof_coordinates[1]-dof_coordinates[0])*(grad_ref_at_quad@grad_ref_at_quad.T)                   
    A[np.ix_(dofIndices,dofIndices)]+=cell_matrix


   #-----assemble right hand side vector   
    local_rhs=np.zeros(p+1)
    for q in range(Quad.nq):
        nodal_ref_at_quad=np.array([1-Quad.xhat[q],Quad.xhat[q]])
        local_rhs+=Quad.what[q]*(dof_coordinates[1]-dof_coordinates[0])*\
        rhs_func(Quad.xhat[q]*(dof_coordinates[1]-dof_coordinates[0])+\
                 dof_coordinates[0])*nodal_ref_at_quad
    F[np.ix_(dofIndices)]+=local_rhs

#----boundary dof------
    
for i in range(len(DOF.dof_bnd)):
    uh[DOF.dof_bnd[i]]=exact_solution(DOF.coordinates[DOF.dof_bnd[i]])    
    
    
F=F-A.dot(uh)     

#----interior dof------ 
uh[DOF.dof_int]=np.linalg.solve(A[np.ix_(DOF.dof_int,DOF.dof_int)],F[DOF.dof_int])

for i in range(DOF.n_dof):
    u[i]=exact_solution(DOF.coordinates[i])

error=u-uh

u_l2=LA.norm(u)
error_l2=LA.norm(error)
relative=error_l2/u_l2
print("relative error=:",relative)



#-------visualize the solutions--------

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.plot(DOF.coordinates, u)
plt.title("exact solution u")
plt.subplot(132)
plt.plot(DOF.coordinates, uh)
plt.title("numerical solution uh")
plt.subplot(133)
plt.plot(DOF.coordinates, error)
plt.title("error")
plt.show()
