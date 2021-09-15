import numpy as np 
import scipy
from scipy import spatial, linalg, io   





# Square grid boundaries
grid_min = 0
grid_max = 25
spacing = (grid_max-grid_min)/(grid_max)

# Number of cells the relay movement area is divided into (rows, columns)
rMap_RowCells = 20
rMap_ColCells = 20



# Channel Hyperparameters
ell= 2.3
rho= 3

P_S = 10**((45-30)/10) #10**((35-30)/10) #source power % dbm to power  
P_R = 10**((55-30)/10) #10**((45-30)/10) #total relay budget power 
sigmaSQ = 1
sigma_xiSQ = 3#10  # multipath variance
sigma_DSQ = 1
etaSQ = 6#20 #(10**(7.7/10))**2 #60 # Shadowing Power    
c1 = 1.2#1# Correlation Distance
c2 = 0.6#1 #120   # Correlation Time
c3 = 0.5  # Base Station Correlation

mstep=1
numEpisodes = 150
numSlots = 400

numRelays = 3




class GridWorld(object):
    def __init__(self, m, n):
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        self.totalGridcells=m*n
        self.stateSpace = [i for i in range(self.totalGridcells)] 
        self.actionSpace = {  'NW':[-1,-1], 'N':[-1,0],   'NE':[-1,1],  
                              'W':[0,-1],   'Stay':[0,0], 'E':[0,1],
                              'SW':[1,-1],  'S':[1,0],    'SE': [1,1]}
        self.candidateActions = ['NW',   'N',    'NE',
                                'W',    'Stay', 'E', 
                                'SW',   'S',    'SE']
        self.actionindex = {  'NW':0, 'N':1,   'NE':2,  
                              'W':3,   'Stay':4, 'E':5,
                              'SW':6,  'S':7,    'SE': 8}
        self.num_actions=len(self.candidateActions)
        

    def setState(self):
    
        print('------------------------------------------')
        grid_bound = np.linspace(grid_min, rMap_RowCells, rMap_RowCells+1)
        grid_X_temp, grid_Y_temp = np.meshgrid(grid_bound,grid_bound)
        grid_X=np.flipud(grid_Y_temp)
        grid_Y=grid_X_temp        

        # All possible relay positions of grid
        relay_pos_X = np.array(grid_X-(spacing/2))
        relay_pos_X = np.delete(relay_pos_X, -1, axis=0)
        relay_pos_X = np.delete(relay_pos_X, -1, axis=1)

        relay_pos_Y = np.array(grid_Y-(spacing/2))
        relay_pos_Y = np.delete(relay_pos_Y, -1, axis=0)
        relay_pos_Y = np.delete(relay_pos_Y, 0, axis=1)

        # Store all grid coordinates of numbered cell (row index corresponds to grid cell number)
        All_gridcords = np.zeros([self.totalGridcells,2]) 
        for grid_cell in range(self.totalGridcells):
            All_gridcords[grid_cell,:] = np.unravel_index(grid_cell,(self.m,self.n)) + np.array([spacing/2, spacing/2])

        # Time lag matrix
        T = np.zeros([numSlots,numSlots])
        for k in range(numSlots): 
            for l in range(numSlots):
                T[k,l] = np.exp(-np.abs(k-l)/c2) #time lag
        
        # grid cell distances
        dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(All_gridcords, 'euclidean'))

        # Create the global covariance matrix
        C = np.zeros([self.totalGridcells, self.totalGridcells]) 
        for i in range(self.totalGridcells):
            for j in range(self.totalGridcells):
                C[i,j]= etaSQ*np.exp(-dist[i,j]/c1)

        del dist
        
        # Source & Destination positions
        S_pos = np.zeros([1, 2]) 
        D_pos = np.zeros([1, 2]) 

        S_pos[0,:] = np.array([24, 10])         
        D_pos[0,:] = np.array([-3, 10])

        S_cord = S_pos + np.array([spacing/2, spacing/2]) 
        D_cord = D_pos + np.array([spacing/2, spacing/2]) 

        C_SD = np.zeros([2*self.totalGridcells, 2*self.totalGridcells])
        C_SD = np.vstack((np.hstack((C,C*np.exp(-np.linalg.norm(S_cord-D_cord)/c3))), np.hstack((C*np.exp(-np.linalg.norm(S_cord-D_cord)/c3),C)) ))

        kappa = np.exp(-1/c2)
        C_SD_chol = scipy.linalg.cholesky( (1-(kappa**2))*C_SD) 


        # Create the path loss matrices for WHOLE grid
        pathlossF = np.zeros([self.m,self.n]) 
        pathlossG = np.zeros([self.m,self.n])        

        for i in range(self.totalGridcells):
            grid_pos = np.unravel_index(i, (self.m,self.n))  
            grid_cord = np.array(grid_pos) + np.array([spacing/2, spacing/2]) 

            pathlossF[grid_pos[0], grid_pos[1]] = -ell*10*np.log10(np.linalg.norm(grid_cord - S_cord)) + rho
            pathlossG[grid_pos[0], grid_pos[1]] = -ell*10*np.log10(np.linalg.norm(grid_cord - D_cord)) + rho

        return T, C_SD, C_SD_chol, kappa, pathlossF, pathlossG, grid_X, grid_Y, S_cord, D_cord
    




def VI_local(f_S, f_D):
    
    VI_numerator= P_R* P_S* (np.abs(f_S)**2) * (np.abs(f_D)**2)
    VI_denominator= (P_S*sigma_DSQ*(np.abs(f_S)**2)) + (P_R*sigmaSQ*(np.abs(f_D)**2)) + (sigmaSQ*sigma_DSQ)
    VI_opt= VI_numerator / VI_denominator #individual SINR
    
    return VI_opt
    

def withinGrid(x, y): 
    if ((x >= grid_min) and (x < rMap_RowCells) and (y >= grid_min) and (y < rMap_ColCells)): 
        return True
    else: 
        return False


def Perfect_CSI(pathlossF_S, pathlossF_D, C_SD, C_SD_chol, kappa):

    
    f_Sphase = np.exp(1j*2*np.pi*np.random.uniform(0,1,(rMap_RowCells, rMap_ColCells, numSlots+1)))
    f_Dphase = np.exp(1j*2*np.pi*np.random.uniform(0,1,(rMap_RowCells, rMap_ColCells, numSlots+1)))

    f_Smaps = np.zeros([rMap_RowCells, rMap_ColCells,numSlots+1], dtype=complex)
    f_Dmaps = np.zeros([rMap_RowCells, rMap_ColCells,numSlots+1], dtype=complex)


    # Generate the spatiotemporal process     
    beta_t = (np.random.randn(C_SD[0,:].shape[0]) @ scipy.linalg.cholesky(C_SD) ).T # initial shadowing term at time t=0 (X_0 initial)

    for i in range(numSlots+1):
        #(Innovation) Generate the driving noise of the AR process
        W_t = (np.random.randn(C_SD[0,:].shape[0]) @ C_SD_chol ).T
        beta_t = kappa * beta_t + W_t #Update the shadowing term for time t
        # Add multipath fading...
        temp = np.array(beta_t + np.sqrt(sigma_xiSQ)*np.random.randn(C_SD[0,:].shape[0]).T)
        temp_S = temp[0:int(temp.shape[0]/2)] # "shadowing + multipath" term (in dB) of source to "points in grid" channels
        temp_D = temp[int(temp.shape[0]/2):] #shadowing + multipath term (in dB) of "points in grid" to destination channels
        
        #add the respective path-loss
        Temp_F_Smaps = pathlossF_S + np.reshape(temp_S, (rMap_RowCells, rMap_ColCells) )#"F_S" channels (in dB) from source to "grid cell" 
        Temp_F_Dmaps = pathlossF_D + np.reshape(temp_D, (rMap_RowCells, rMap_ColCells) )#"F_D" channels (in dB) from "grid cell" to destination 

        # Grid channels in linear scale
        f_Smaps[:,:,i] = (10**(Temp_F_Smaps/20)) * f_Sphase[:,:,i]
        f_Dmaps[:,:,i] = (10**(Temp_F_Dmaps/20)) * f_Dphase[:,:,i]

    return f_Smaps, f_Dmaps




def Qtable_initialization(stateSpace, actionSpace, candidateActions):

    # read_dictionary = np.load('Qsamples.npy', allow_pickle='TRUE')
    
    Qtable_initial = {}
    validMoves = [[] for i in   stateSpace]
    count = -1
    for state in stateSpace:
        for action in candidateActions:
            count += 1
            temp_relay_pos = np.unravel_index(state, (rMap_RowCells, rMap_ColCells)) + np.array([actionSpace[action]])            
            if withinGrid(temp_relay_pos[0,0], temp_relay_pos[0,1]): # If given action leads to a valid grid position
                validMoves[state].append(action)
                # Qtable_initial[state, action] = read_dictionary[count]
                Qtable_initial[state, action] = 0
 
    return Qtable_initial, validMoves



