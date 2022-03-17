"""  		   	  			  	 		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			  	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			  	 		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			  	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			  	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			  	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			  	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			  	 		  		  		    	 		 		   		 		  
or edited.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		   	  			  	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			  	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			  	 		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
Student Name: Alonso Torres-Hotrum		   	  			  	 		  		  		    	 		 		   		 		  
GT User ID: atorreshotrum3		   	  			  	 		  		  		    	 		 		   		 		  
GT ID: 903475423   			   	  			  	 		  		  		    	 		 		   		 		  
"""  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
import numpy as np  		   	  			  	 		  		  		    	 		 		   		 		  
import random as rand  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
class QLearner(object):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		   	  			  	 		  		  		    	 		 		   		 		  
        self.num_actions = num_actions  
        self.num_states = num_states		   	  			  	 		  		  		    	 		 		   		 		  
        self.s = 0  		   	  			  	 		  		  		    	 		 		   		 		  
        self.a = 0  		   	  
        self.Q = np.zeros((self.num_states, self.num_actions))	
        self.alpha = alpha 
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna		

        self.T = np.empty((num_states, num_actions, num_states))
        self.Tc = np.empty_like(self.T) 
        self.Tc.fill(0.0001)	 	

        self.R = np.zeros((num_states, num_actions))	  		  	

        self.history = {}    	 		 		   		 		  

    def author(self):
        return 'atorreshotrum3'

    def querysetstate(self, s):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        self.s = s  		   	  			  	 		  		  		    	 		 		   		 		  

        random_number = rand.random() 		  	

        if random_number > self.rar:
            self.a = np.argmax(self.Q[self.s, :])
        else:   		 		  
            self.a = rand.randint(0, self.num_actions-1)  	
        
        if self.verbose: print(f"s = {s}, a = {self.a}")  		 
	  			  	 		  		  		    	 		 		   		 		          	   	  			  	 		  		  		    	 		 		   		 		  
        return self.a  		

  		   	  			  	 		  		  		    	 		 		   		 		  
    def query(self,s_prime,r):  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		   	  			  	 		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			  	 		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			  	 		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			  	 		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			  	 		  		  		    	 		 		   		 		  
        """  		  
        if self.verbose: print(f"s = {s_prime}, a = {self.a}, r={r}") 

        # Log state and action into history
        if self.s in self.history.keys():
            if self.a not in self.history[self.s]:
                self.history[self.s].append(self.a)
        else:
            self.history[self.s] = [self.a]

        self.R[self.s, self.a] = r

        # Update Q Table
        self.Q[self.s, self.a] = self.Q[self.s, self.a] + self.alpha * (r + self.gamma * np.max(self.Q[s_prime, :]) - self.Q[self.s, self.a])
        
        # If dyna not zero
        if self.dyna != 0:

            # Update the Tc matrix
            self.Tc[self.s, self.a, s_prime] += 1

            # Update T matrix
            self.T = (self.Tc / np.sum(self.Tc[self.s, self.a]))

            # Update R' matrix
            R_prime = (1-self.alpha) * self.R + self.alpha * r

            # Hallucinate
            for i in range(self.dyna):

                # Get random (previously seen) action and state
                temp_s = rand.choice(list(self.history.keys()))
                self.a = rand.choice(list(self.history[temp_s]))

                # Get s' from T matrix
                temp_s_prime = np.argmax(self.T[temp_s, self.a])

                # Get reward
                r = R_prime[temp_s, self.a]

                # Update Q
                self.Q[temp_s, self.a] = self.Q[temp_s, self.a] + self.alpha * (r + self.gamma * np.max(self.Q[temp_s_prime, :]) - self.Q[temp_s, self.a])

        # Update s
        self.s = s_prime

        
        random_number = rand.random() 		  	

        if random_number > self.rar:
            self.a = np.argmax(self.Q[self.s, :])
        else:   		 		  
            self.a = rand.randint(0, self.num_actions-1)  	


        # Update rar
        self.rar *= self.radr	   	  			  	 		  		  		    	 		 		   		 		  
                	   	  			  	 		  		  		    	 		 		   		 		  
        return self.a  		   	  			  	 		  		  		    	 		 		   		 		  
  		   	  			  	 		  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			  	 		  		  		    	 		 		   		 		  
    ql = QLearner(num_states=2, num_actions = 3,dyna = 3)
    ql.query(1, 3)
    
    	   	  			  	 		  		  		    	 		 		   		 		  
