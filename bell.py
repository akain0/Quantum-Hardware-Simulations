from secrets import randbits
import numpy as np       

class bell_ineq_base(object):
    """ Calculate Bell inequalities from randomly generated hidden variables """

    # constructor function initializing the RNG
    def __init__(self):
        self.local_rng = np.random.default_rng(randbits(128))

    # function to generate hidden variables array (3xN)
    def generate_hidden_variables(self,N):
        return self.local_rng.choice([-1,1],(3,N))
    
    # utility function to get a numerical representation for things like:
    # x.calculate_PXY(x.generate_hidden_variables(100),"+A","+B")
    def axis_text_to_num(self,text):
        text_up = text.upper()
        if text_up == "+A":
            return np.array([1,0,0])
        elif text_up == "-A":
            return np.array([-1,0,0])
        elif text_up == "+B":
            return np.array([0,1,0])
        elif text_up == "-B":
            return np.array([0,-1,0])
        elif text_up == "+C":
            return np.array([0,0,1])
        elif text_up ==  "-C":
            return np.array([0,0,-1])

    # function to calculate P(X,Y), where X and Y are strings such as "+A", or "-C", etc.
    def calculate_PXY(self,hidden_vars,axis1,axis2):
        count = 0
        axis_idx = {
            "+A": 0,
            "-A": 0,
            "+B": 1,
            "-B": 1,
            "+C": 2,
            "-C": 2
        }
        
        axis_1 = self.axis_text_to_num(axis1)
        axis_2 = self.axis_text_to_num(axis2)
        pairs_arr = [[] for _ in range(hidden_vars.shape[1])]
        
        for axis in hidden_vars:
            for elem in range(len(axis)):
                    pairs_arr[elem].append(int(axis[elem]))    

        if axis1 == axis2:
            joint_prob_axis = axis_1
        else:
            joint_prob_axis = axis_1+axis_2

        pairs_arr = np.array(pairs_arr)
        n_pairs = len(pairs_arr)
        
        axis_entry_1 = axis_idx[axis1]
        axis_entry_2 = axis_idx[axis2]

        if axis_entry_1 == axis_entry_2:
            for pair in enumerate(pairs_arr):
                if pair[axis_entry_1] == joint_prob_axis[axis_entry_1]:
                    count+=1
            return count/n_pairs
        
        for i, pair in enumerate(pairs_arr):
            if pair[axis_entry_1] == joint_prob_axis[axis_entry_1] and pair[axis_entry_2] == joint_prob_axis[axis_entry_2]:
                count+=1
        
        return count/n_pairs
        
    # function to return the bell inquality P(+A,-B) + P(+B,-C) - P(+A,-C)
    # which should be greater than zero in every case
    def bell_inequality(self,hidden_vars):
        return self.calculate_PXY(hidden_vars, '+A', '-B') + self.calculate_PXY(hidden_vars, '+B', '-C') - self.calculate_PXY(hidden_vars, '+A', '-C')

class bell_ineq(bell_ineq_base):
    """ Calculate Bell inequalities from randomly generated hidden variables """


if __name__ == '__main__':
    bell_test = bell_ineq()

    print(bell_test.calculate_PXY(bell_test.generate_hidden_variables(10), "+A", "+A"))
