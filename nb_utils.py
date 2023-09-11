import time
import numpy as np
import matplotlib.pyplot as plt

# A simple class for timing contexts
class IndividualTimer:
    def __init__(self, name, n_iter=None):
        self.name = name
        self.t_start = None
        self.t_end = None
        self.n_iter = n_iter
        self.t_elapsed = None
        self.sec_per_iter = None
        self.iter_per_sec = None
        
        
    def __enter__(self):
        self.t_start = time.time()
        
        
    def __exit__(self, type, value, traceback):
        self.t_end = time.time()
        self.t_elapsed = self.t_end - self.t_start
        
        self.n_iter = 1 if self.n_iter is None else self.n_iter
        self.sec_per_iter = self.t_elapsed / self.n_iter
        self.iter_per_sec = self.n_iter / self.t_elapsed 
        
        self.print_timing()
        
        
    def print_timing(self):
        
        if self.n_iter == 1:
            print(f'Total time elapsed for timer {self.name}: {self.t_elapsed:.3f} seconds')
        else:
            print(f'Timings ({self.name}):')
            print(f'  {self.n_iter} iterations in {self.t_elapsed:.3f} seconds')
            print(f'  {self.iter_per_sec:.4f} iter/sec')
            print(f'  {self.sec_per_iter:.5f} sec/iter')
            
            
# This is a Singleton timer wrapper class, which will track all timers in a simple way
class SimpleTimers:
    all_timers = {}
    
    @classmethod
    def new(cls, name, n_iter=None):
        cls.all_timers[name] = IndividualTimer(name, n_iter)
        return cls.all_timers[name]
        
        
    @classmethod
    def get_timer(cls, name):
        return cls.all_timers[name]

        
    @classmethod
    def get_iter_per_sec(cls, name):
        return cls.all_timers[name].iter_per_sec
    
    @classmethod
    def get_sec_per_iter(cls, name):
        return cls.all_timers[name].sec_per_iter
    
    @classmethod
    def get_total_time(cls, name):
        # If we knew only one iter was run, this method name is clearer
        return cls.all_timers[name].t_elapsed
    


def show_complex_mesh(zmesh_real, zmesh_imag):
    # Assume square
    EPS = 1e-8
    xsz,ysz = zmesh_real.shape
    
    fig,axs = plt.subplots(1,2, figsize=(6,4))
    axs[0].imshow(zmesh_real)
    axs[0].set_title('zmesh_real')
    axs[1].imshow(np.flipud(zmesh_imag))
    axs[1].set_title('zmesh_imag')
    for i in range(2):
        axs[i].set_xticks(np.arange(0, xsz+1, ysz//4))
        axs[i].set_xticklabels(np.arange(-2, 2+EPS, 1))
        axs[i].set_yticks(np.arange(0, xsz+1, ysz//4))
        axs[i].set_yticklabels(np.arange(2, -2-EPS, -1))
        axs[i].set_xlabel('Real Axis')
        axs[i].set_ylabel('Imaginary Axis')
    fig.tight_layout(pad=0.4)
    fig.suptitle('Real & Imaginary Parts of Complex Plane')
    
    
def relabel_axes(ax, nr=768, rmin=-2, rmax=2, ni=768, imin=-2, imax=2):
    r_width = rmax - rmin
    new_rticks = np.arange(0, nr+1, nr//8, dtype=np.int32).tolist()
    new_rticklabels = [f'{r_width * t / nr + rmin:.1f}' for t in new_rticks]
    ax.set_xticks(new_rticks)
    ax.set_xticklabels(new_rticklabels)
    ax.set_xlabel('Real Axis')
    
    i_width = imax - imin
    new_iticks = np.arange(0, ni+1, ni//8, dtype=np.int32).tolist()
    new_iticklabels = [f'{i_width * t / ni + imin:.1f}' for t in new_iticks][::-1]
    ax.set_yticks(new_iticks)
    ax.set_yticklabels(new_iticklabels)
    ax.set_ylabel('Imaginary Axis')
    
    
def plot_bars_speedup(labels, values, title):
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values)

    # Add labels above the bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Add title and axis labels (optional)
    plt.title(title)
    plt.ylabel('Relative Speed')
    plt.ylim(0, max(values)*1.1)

    # Show the plot
    plt.show()