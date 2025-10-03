import numpy as np              
from scipy import constants     
import matplotlib.pyplot as plt
from typing import Optional

class josephson_base(object):
    """ Base features for Josephson junction solver """

    # constructor function initializing the Josephson constant
    def __init__(self):
        self.josconst = constants.pi*2*483597.84841698*1E9 # 2e/hbar based on exact SI defined value in 1/(V*s)

    # function to load circuit parameters
    def load_params(self,imax,cap,res,biascurr):
        self.Ejunc = imax/self.josconst
        self.cap = cap
        self.res = res
        self.biascurr = biascurr

    # function to initialize junction phase and charge
    def init_junction(self,phase,charge):
        self.phase = phase
        self.charge = charge

    # function to calculate the voltage
    def calc_voltage(self):
        self.voltage = 0
    
    # function to calculate the currents
    def calc_currents(self):
        self.rescurr = 0
        self.supercurr = 0
        self.capcurr = 0

    # function to execute a single time step
    def execute_time_step(self,duration):
        pass
    
    # function to simulate a duration T with N steps (iterates over the single step version)
    def evolve_for(self,T,N):
        self.calc_voltage()
        duration = T/N
        self.recorded_voltage = np.zeros((N+1))
        self.recorded_phase = np.zeros((N+1))
        self.recorded_time = np.zeros((N+1))
        for t in range(N):
            self.recorded_voltage[t] = self.voltage
            self.recorded_phase[t] = self.phase
            self.recorded_time[t] = t*duration
            self.execute_time_step(duration)
        self.recorded_voltage[N] = self.voltage
        self.recorded_phase[N] = self.phase
        self.recorded_time[N] = N*duration

            
    # utility function using matplotlib.pyplot for voltage
    def plot_voltage(self, type = "", title=""):
        plt.title(label=title)
        plt.plot(self.recorded_time, self.recorded_voltage, label=type)

    # utility function using matplotlib.pyplot for phase
    def plot_phase(self, type = "", title=""):
        plt.title(label=title)
        plt.plot(self.recorded_time, self.recorded_phase, label=type)

    # generate oscillating underdamped case
    def do_underdamped_unstable(self,show):
        pass
    
    # generate stable underdamped case
    def do_underdamped_stable(self,show):
        pass

    # generate oscillating overdamped case
    def do_overdamped_unstable(self,show):
        pass

    # generate stable overdamped case
    def do_overdamped_stable(self,show):
        pass

    def find_critical_voltage(self):
        return 0
    
    def find_spike_freq(self):
        return 0
    
    def find_avg_voltage(self):
        return 0

class josephson_solver(josephson_base):
    """ Solve tilted washboard model for a Josephson junction """

    def calc_voltage(self):
        self.voltage = self.charge/self.cap

    def calc_currents(self):
        self.rescurr = self.voltage/self.res
        self.supercurr = self.Ejunc *self.josconst* np.sin(self.phase)
        self.capcurr = self.biascurr - self.rescurr - self.supercurr

    def execute_time_step(self,duration):
        self.calc_voltage()
        dphi_dt = self.josconst * self.voltage
        self.phase = self.phase + dphi_dt * duration
        
        self.calc_currents()
        
        dcharge_dt = self.capcurr
        self.charge = self.charge + dcharge_dt * duration

    def do_underdamped_unstable(self,show):
        self.load_params(imax=55e-6, cap=1e-12, res=16, biascurr=40e-6)
        initial_voltage = 160e-6
        initial_charge = initial_voltage * self.cap
        self.init_junction(phase=0, charge=initial_charge)
        self.evolve_for(T=1e-10, N=90000)
        if show:
            self.plot_voltage("Underdamped Unstable")
            plt.show()

    def do_underdamped_stable(self,show):
        self.load_params(imax=55e-6, cap=1e-12, res=16, biascurr=40e-6)
        initial_voltage = 80e-6
        initial_charge = initial_voltage * self.cap
        self.init_junction(phase=0, charge=initial_charge)
        self.evolve_for(T=1e-10, N=90000)
        if show:
            self.plot_voltage(title="Underdamped Stable Potential at Junction over Time")
            plt.show()

    def superimpose_underdamped_cases(self):
        # unstable
        self.load_params(imax=55e-6, cap=1e-12, res=16, biascurr=40e-6)
        initial_voltage = 160e-6
        initial_charge = initial_voltage * self.cap
        self.init_junction(phase=0, charge=initial_charge)
        self.evolve_for(T=1e-10, N=90000)
        self.plot_voltage(type="Underdamped Unstable")
        # stable
        self.load_params(imax=55e-6, cap=1e-12, res=16, biascurr=40e-6)
        initial_voltage = 80e-6
        initial_charge = initial_voltage * self.cap
        self.init_junction(phase=0, charge=initial_charge)
        self.evolve_for(T=1e-10, N=90000)
        self.plot_voltage(type="Underdamped Stable", title="Underdamped Stable vs Unstable Potential at Junction over Time")
        plt.legend()
        plt.show()

    def do_overdamped_unstable(self,show):
        self.load_params(imax=55e-6, cap=1e-12, res=1.6, biascurr=56e-6)
        initial_voltage = 100e-6
        initial_charge = initial_voltage * self.cap
        self.init_junction(phase=0, charge=initial_charge)
        self.evolve_for(T=1e-9, N=90000)
        if show:
            self.plot_voltage(title="Overdamped Unstable Potential at Junction over Time")
            plt.show()
           

    def do_overdamped_stable(self,show):
        self.load_params(imax=55e-6, cap=1e-12, res=1.6, biascurr=54e-6)
        initial_voltage = 100e-6
        initial_charge = initial_voltage * self.cap
        self.init_junction(phase=0, charge=initial_charge)
        self.evolve_for(T=1e-9, N=90000)
        if show:
            self.plot_voltage(title="Overdamped Stable Potential at Junction over Time")
            plt.show()

    def find_critical_voltage(self):
        underdamped_stable_starting_voltage = 80e-6
        underdampled_unstable_starting_voltage = 160e-6
        n_linspaces = 5000
        test_voltages = np.linspace(underdamped_stable_starting_voltage, underdampled_unstable_starting_voltage, n_linspaces) # as no. of linear spaces -> /inf, solution converges
        last_stable = 0
        first_unstable = 0
        for v_test in test_voltages:
            josephson = josephson_solver()
            josephson.load_params(imax=55e-6, cap=1e-12, res=16, biascurr=40e-6)
            initial_charge = v_test * josephson.cap
            josephson.init_junction(phase=0, charge=initial_charge)
            josephson.evolve_for(T=1e-10, N=90000)
            
            final_avg = np.mean(np.abs(josephson.recorded_voltage[-1000:]))
            initial_avg = np.mean(np.abs(josephson.recorded_voltage[1000:2000]))
            
            if final_avg > initial_avg * 1.5:
                status = "UNSTABLE"
                first_unstable = v_test
                break
            else:
                status = "STABLE"
                last_stable = v_test

        critical_voltage = round(last_stable, 9)
        return critical_voltage
    
    def find_spike_freq(self):
        # spike frequency (in GHz) for overdamped unstable case: f = 1/T, T = period of oscillation
        self.do_overdamped_unstable(show=False)
        voltage_vs_time_dict = {self.recorded_time[i]:self.recorded_voltage[i] for i in range(len(self.recorded_voltage))}
        sorted_items_asc = sorted(voltage_vs_time_dict.items(), key=lambda item: item[1])
        # f = 1/T = 1/(peak_1_time - peak_2_time) 
        peak_1_time = sorted_items_asc[0][0]
        peak_2_time = sorted_items_asc[1][0]
        freq_in_Hz = 1/(peak_1_time - peak_2_time)
        freq_in_GHz = round(freq_in_Hz * 1e-9, 3)

        return freq_in_GHz
    
    def find_avg_voltage(self):
        # average voltage also for overdamped unstable case
        self.do_overdamped_unstable(show=False)
        avg_voltage = round(np.mean(self.recorded_voltage), 8)
        return avg_voltage

if __name__ == '__main__':
    josephson = josephson_solver()
    josephson.superimpose_underdamped_cases()

    
