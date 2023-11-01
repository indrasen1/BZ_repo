import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import timeit
from IPython.display import display, clear_output
from matplotlib import animation
import matplotlib as mpl
import sys

mpl.rcParams['animation.ffmpeg_path'] = r'/Users/indrasen/Desktop/ffmpeg'

class BZ_solver():
    '''
    20230717: Indrasen Bhattacharya
    Class to implement the BZ solver in BZ sandbox notebook
    No inheritance
    Each instance variable represents a given system and consists of:
    -> Initialization: either constructed shape or png input, random scale, image size
    -> Forcing function: initialization points, species to be reset
    -> System parameters: rate constants, diffusion constants, length of time, time step
    -> Member functions:
        -> Constructor (__init__): decide method to initialize and create the arrays
        -> Square, circle and heart construction shapes
        -> Load png
        -> Forcing function
        -> BZ propagator (returns derivative)
        -> Full Runge-Kutta 
        -> Notebook display function
        -> Function to convert to animatioN
    '''
    
    def __init__( self, params=None ):
        #constructor
        
        #if parameters are not provided, use the defaults:
        #hardcoded based on an existing working example
        if params is None:
            params = {}
            params['imgSize'] = 201
            params['conc_scale'] = 0.05
            
            params['alpha'] = 2.5
            params['beta'] = 2
            params['gamma'] = 2
            
            params['dA'] = 1
            params['dB'] = 1
            params['dC'] = 1
            
            params['isBlurred'] = True
            params['loadPng'] = False
            params['shape'] = 'heart'
            
            params['tSteps'] = 1000
            params['dt'] = 0.1
            
            #select the forced species and time steps
            params['species'] = [0]
            params['t_arr'] = [0]
            #params['t_arr'] = [0, 500//3, 1000//3, 1500//3, 2000//3]
            
        
        #initialize basic parameters
        self.n = params['imgSize']
        self.conc_scale = params['conc_scale']
        self.blurred = params['isBlurred']
        
        #perform initialization using the sigmoid
        #A, B, C represent the three species
        A = np.random.randn( self.n, self.n )
        B = np.random.randn( self.n, self.n )
        C = np.random.randn( self.n, self.n )
        
        #if blurring, perform simple convolution with a 3*3 blurring filter
        #this is hard-coded for now
        if self.blurred:
            conv_filter = np.array([[0.3, 0.5, 0.3], [0.5, 1, 0.5], [0.3, 0.5, 0.3]])
            conv_filter = conv_filter / np.linalg.norm(conv_filter)
            A = sp.ndimage.convolve(A, conv_filter, mode='reflect')
            B = sp.ndimage.convolve(B, conv_filter, mode='reflect')
            C = sp.ndimage.convolve(C, conv_filter, mode='reflect')
        
        #perform sigmoid to restrict to a reasonable range
        A = self.conc_scale * self.sigmoid(A, np.std(A) )
        B = self.conc_scale * self.sigmoid(B, np.std(B) )
        C = self.conc_scale * self.sigmoid(C, np.std(C) )
        
        self.X_init = np.stack([A, B, C], axis=-1)
        self.tSteps = params['tSteps']
        self.dt = params['dt']
        x_shape = np.shape( self.X_init )
        self.X_arr = np.zeros( np.concatenate([x_shape, np.array([self.tSteps])]) )
        
        #initialize rate constants
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.gamma = params['gamma']
        
        #initialize diffusion constants
        self.dA = params['dA']
        self.dB = params['dB']
        self.dC = params['dC']
        
        if params['shape'] == 'heart':
            
            if not 'heartParams' in params:
                params['heartParams'] = {}
                params['heartParams']['a'] = params['imgSize']//4
                params['heartParams']['b'] = params['imgSize']//4
            
            P_loc, S_loc = self.heart( params['heartParams']['a'], params['heartParams']['b'] )
            
        elif params['shape'] == 'square':
            
            if not 'squareParams' in params:
                params['squareParams'] = {}
                params['squareParams']['s'] = params['imgSize']//4
                
            P_loc, S_loc = self.square( params['squareParams']['s'] )
            
        elif params['shape'] == 'circle':
            
            if not 'circParams' in params:
                params['circParams'] = {}
                params['circParams']['r'] = params['imgSize']//4
                params['circParams']['c'] = None
                
            P_loc, S_loc = self.circle( params['circParams']['r'], params['circParams']['c'] )
            
        elif params['loadPng'] is True:
            
            #default img name in case this is absent
            if not 'imgName' in params:
                params['imgName'] = '20230714_testImg.png'
            
            P_loc, S_loc = self.loadPng( params['imgName'] )
            
        
        elif params['loadPng_allSpecies'] is True:
            
            P_loc, S_loc = self.loadPng_3species( params['imgNameList'] )
            
        
        #if the species are not specified
        #stack along species dimension
        #this allows for single species specification as well as other cases
        #repeat thrice for the maximum possible 3 species in the chemistry
        if P_loc.ndim == 2:
            P_loc = np.repeat( P_loc[..., np.newaxis], 3, axis=2 )
            S_loc = np.repeat( S_loc[..., np.newaxis], 3, axis=2 )
            
            
        #write code snippet here to initialize the forcing functions
        self.t_arr = params['t_arr']
        if ( any ( np.array(self.t_arr) >= self.tSteps ) ):
            print('AT LEAST ONE OF THE FORCING FUNCTION TIME STEPS IS INVALID')
            self.t_arr = [0]
            
        
        #choose only the first specie
        if not 'species' in params:
            self.species = [0]
        else:
            self.species = params['species']
        
        
        #P is the multiplicative forcing function
        #S is the additive forcing function
        self.P = np.ones_like( self.X_arr )
        self.S = np.zeros_like( self.X_arr )
        
        
        for t in self.t_arr:
            for s in self.species:
                self.P[..., s, t] = P_loc[..., s]
                self.S[..., s, t] = S_loc[..., s]
        
        
        #define the Laplacian for the discrete diffusion term
        self.L = np.array([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]])
    
    def displForcing( self, tLoc = 0 ):
        #helper function to display the forcing function at a particular time step
        #default time step is 0
        #forcing function displayed for all forced species
        nS = len(self.species)
        print('NUMBER OF FORCED SPECIES ' + str(nS) )
        
        fig, ax = plt.subplots( nrows=nS, ncols=2, figsize=(12, 6*nS), squeeze=False )
        print('DISPLAYING FORCING FUNCTION FOR EACH SPECIES IN ORDER')
        
        for iS in range(nS):
            #print('DISPLAYING FORCING FUNCTION FOR SPECIES ' + str(self.species[iS]))
            h0 = ax[iS, 0].imshow( self.P[..., self.species[iS], tLoc], cmap='gray' )
            h1 = ax[iS, 1].imshow( self.S[..., self.species[iS], tLoc], cmap='gray' )
            fig.colorbar( h0, ax=ax[iS, 0], fraction=0.046, pad=0.04 )
            fig.colorbar( h1, ax=ax[iS, 1], fraction=0.046, pad=0.04 )
            
            time.sleep(0.01)
            
            
        
        plt.show()
        
        #h0 = ax[0].imshow( self.P[..., self.species[0], self.t_arr[0] ] )
        #h1 = ax[1].imshow( self.S[..., self.species[0], self.t_arr[0] ] )
        
        #fig.colorbar(h0, ax=ax[0])
        #fig.colorbar(h1, ax=ax[1])
        
        
    def propagator( self, X_f ):
        #helper function to return the derivative based on the current state
        
        alpha_ = self.alpha
        beta_  = self.beta
        gamma_ = self.gamma
        
        dA_ = self.dA
        dB_ = self.dB
        dC_ = self.dC
        
        L = self.L
        
        dA = X_f[..., 0] * ( alpha_ * X_f[..., 1] - gamma_ * X_f[..., 2] ) + dA_ * sp.ndimage.convolve(X_f[..., 0], L, mode='wrap')
        dB = X_f[..., 1] * ( beta_ * X_f[..., 2] - alpha_ * X_f[..., 0] ) + dB_ * sp.ndimage.convolve(X_f[..., 1], L, mode='wrap')
        dC = X_f[..., 2] * ( gamma_ * X_f[..., 0] - beta_ * X_f[..., 1] ) + dC_ * sp.ndimage.convolve(X_f[..., 2], L, mode='wrap')
        
        dX_f = np.stack([dA, dB, dC], axis=-1)
        
        return dX_f
    
    
    def solver_RK4( self ):
        #run the solver and populate the full array
        #most time consuming aspect of the problem
        #is it necessary to retain a generic forcing function?
        
        start = timeit.timeit()
        
        self.X_arr[..., 0] = self.X_init
        
        dt = self.dt
        t_steps = self.tSteps
        
        for t in range(t_steps):
            
            if t==0:
                continue
                
            #apply the forcing function on the prior step
            self.X_arr[..., t-1] = self.X_arr[..., t-1] * self.P[..., t-1] + self.S[..., t-1]
            
            #Runge-Kutta propagation of the current step
            
            k1 = self.propagator( self.X_arr[..., t-1] )
            k2 = self.propagator( self.X_arr[..., t-1] + dt*k1/2 )
            k3 = self.propagator( self.X_arr[..., t-1] + dt*k2/2 )
            k4 = self.propagator( self.X_arr[..., t-1] + dt*k3 )
            
            self.X_arr[..., t] = self.X_arr[..., t-1] + dt*(k1 + 2*k2 + 2*k3 + k4)/6
            
        end = timeit.timeit()
        
        print('TOTAL SOLVER TIME: ' + str(start-end) )
        
    
    def displResult( self, species=0, cmapName=None ):
        #function to display the result after propagation
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        if cmapName is None:
            cmapName = 'gray'
        
        for t in range(self.tSteps):
            
            ax.cla()
            f_ = ax.imshow(self.X_arr[..., species, t], cmap=cmapName)
            display(fig)
            #cbar = fig.colorbar(f_, ax, fraction = cbar_frac*im_ratio)
            clear_output(wait=True)
            plt.pause(0.01)
        
    def displTimeProgression( self, loc, tMask=None ):
        #function to display the time progression at a given location
        
        if tMask is None:
            tMax = np.shape(self.X_arr)[-1]
            tMask = np.linspace(0, tMax, tMax)
            
        else:
            tMax = int(np.amax(tMask))
        
        plt.plot( tMask, self.X_arr[loc[0], loc[1], 0, :tMax] )
        plt.plot( tMask, self.X_arr[loc[0], loc[1], 1, :tMax] )
        plt.plot( tMask, self.X_arr[loc[0], loc[1], 2, :tMax] )
        
        plt.title('TIME PROGRESSION')
        plt.legend(['A', 'B', 'C'])
        
        
    def sigmoid( self, x, sig_ ):
        #helper function for initialization
        
        mu = np.mean(x)
        return 1/(1 + np.exp( -(x-mu)/sig_ ) )
    
    
    def square( self, s=51 ):
        #return square
        
        n = self.n
        P_ = np.ones( (n, n) )
        S_ = np.zeros( (n, n) )
        
        if s>=n:
            print('TOO LARGE TO FIT INTO DIMENSIONS')
            return
        
        x1 = n//2 - s//2
        x2 = n//2 + s//2
        
        y1 = n//2 - s//2
        y2 = n//2 + s//2
        
        P_[ x1:x2, y1:y2 ] = 0
        S_[ x1:x2, y1:y2 ] = 1
        
        return P_, S_
    
    
    def circle( self, r = 25, c = None):
        #circle
        
        n = self.n
        
        P_ = np.ones( (n, n) )
        S_ = np.zeros( (n, n) )
        
        if r>=n//2:
            print('TOO LARGE TO FIT INTO DIMENSIONS')
            return
        
        x__ = np.linspace(0, n, n)
        
        if c is None:
            c = (n//2, n//2)
            
        Mx, My = np.meshgrid( x__, x__ )
        
        #equation defining the interior
        #generalize to a few further shapes if desired
        circle = ( (Mx - c[0])**2 + (My - c[1])**2 <= r**2 )
        
        #plt.pcolormesh( Mx, My, circle )
        
        P_[ circle ] = 0
        S_[ circle ] = 1
        
        return P_, S_
    
    
    def heart( self, a = 75, b = 75 ):
        '''
        20230714: Indrasen Bhattacharya
        Fourth order heart shape as per the equation
        (x/a)^2+[y/b-((x/a)^2)^(1/3)]^2=1
        '''
        
        n = self.n
        
        P_ = np.ones( (n, n) )
        S_ = np.zeros( (n, n) )
        
        x__ = np.linspace(0, n, n)
        
        Mx, My = np.meshgrid( x__, x__ )
        
        #fourth order curve equation
        heart = ( ((Mx-n//2)/a)**2 + ( -((My-5*n//9)/b) - (((Mx-n//2)/a)**2)**(1/3) )**2  - 1 < 0 )
        
        P_[ heart ] = 0
        S_[ heart ] = 1 
        
        return P_, S_
    
    def postUpsampler(self, f=5):
        #upsample the image
        #after propagating
        
        X_resampled = sp.signal.resample( self.X_arr, num=f*self.n, axis=0 )
        X_resampled = sp.signal.resample( X_resampled, num=f*self.n, axis=1 )
        
        print('NEW SHAPE OF X_arr: ')
        print(X_resampled.shape)
        
        self.X_arr = X_resampled
        self.n = f*self.n
        
    
    
    def loadPng(self, imgName):
        
        n = self.n
        
        img_loc = plt.imread(imgName)
        sMin = min(img_loc.shape[:2])
        
        img_crop = np.mean( img_loc[:sMin, :sMin, ...], axis=-1 )
        
        #plt.imshow(img_crop[..., 1], cmap='gray')
        
        img_resampled = sp.signal.resample( img_crop, num=n, axis=0 )
        img_resampled = sp.signal.resample( img_resampled, num=n, axis=1)
        
        print(img_crop.shape)
        print(img_resampled.shape)
        
        print('FINAL RESAMPLED AND CROPPED IMAGE:')
        #img_out = np.mean(img_resampled, axis=-1)
        img_out = img_resampled / np.amax(img_resampled)
        
        plt.imshow( img_out, cmap='gray')
        
        P_ = (img_out != 0)
        S_ = img_out
        
        return P_, S_
    
    
    #supply list of names of images
    #one image for each species, in order
    def loadPng_3species(self, imgNames):
        
        n = self.n
        iS = 0
        
        P_ = np.ones( (n, n, 3) )
        S_ = np.zeros( (n, n, 3) )
        
        fig, axs = plt.subplots(nrows = 3, ncols=1, figsize=(6, 18) )
        
        #resample each input image independently
        for localName in imgNames:
            
            img_loc = plt.imread(localName)
            sMin = min(img_loc.shape[:2])
            
            #last axis is RGB color: just average along these before resampling
            img_crop = np.mean( img_loc[:sMin, :sMin, ...], axis=-1 )
            
            img_resampled = sp.signal.resample( img_crop, num=n, axis=0 )
            img_resampled = sp.signal.resample( img_resampled, num=n, axis=1 )
            
            #print(img_crop.shape)
            #print(img_resampled.shape)
            
            axs[iS].set_title('CROPPED IMAGE FOR SPECIES ' + str(iS) )
            #img_out = np.mean(img_resampled, axis=-1)
            img_out = img_resampled / np.amax(img_resampled)
            
            axs[iS].imshow( img_out, cmap='gray')
            
            time.sleep(0.01)
            
            P_[..., iS] = (img_out != 0 )
            S_[..., iS] = img_out 
            
            iS = iS + 1
            if iS == 3:
                plt.show()
                break
        
        plt.show()
                
        return P_, S_
            
    
    
    def generateAnimation( self, species=0, cmapName='RdPu_r', fName='test.mp4' ):
        #function to generate an animation from the fully computed result
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5), layout='compressed')

        fig.set_constrained_layout_pads()

        sLoc = species

        ims = []

        for t in range( np.shape( self.X_arr )[-1] ):
            im = ax.imshow( self.X_arr[..., sLoc, t], animated=True, cmap=cmapName )
            ax.set_axis_off()
            if t==0:
                ax.imshow( self.X_arr[..., sLoc, t], cmap=cmapName )
                ax.set_axis_off()
            ims.append([im])

        #fig.set_tight_layout(tight=True)

        ani = animation.ArtistAnimation( fig, ims, interval=50, blit=True, repeat_delay=1000 )

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='whodat'), bitrate=1800)


        ani.save(filename=fName, writer=writer)
        plt.show()
        
        
if __name__ == 'main':
    import sys
    
    #initialize default parameters based on input string
    
    params_img = {}
    params_img['imgSize'] = 201
    params_img['conc_scale'] = 0.0001

    params_img['alpha'] = 2
    params_img['beta'] = 2
    params_img['gamma'] = 2

    params_img['dA'] = 1
    params_img['dB'] = 1
    params_img['dC'] = 1

    params_img['isBlurred'] = True
    

    params_img['tSteps'] = 1000
    params_img['dt'] = 0.1

    params_img['species'] = [0, 1]
    params_img['t_arr'] = [0, 500//3, 1000//3, 1500//3, 2000//3]

    flag = 1
    
    if sys.argv[1] == 'load hearts':
        
        params_img['loadPng'] = True
        params_img['imgName'] = '20230722_heartAttempt2.png'
        params_img['shape'] = None
    
    elif sys.argv[1] == 'heart':
        
        params_img['loadPng'] = False
        params_img['shape'] = 'heart'
        
    elif sys.argv[1] == 'square':
        
        params_img['loadPng'] = False
        params_img['shape'] = 'square'
        
    elif sys.argv[1] == 'circle':
        
        params_img['loadPng'] = False
        params_img['shape'] = 'circle'
        
    else:
        
        print('INVALID PROMPT')
        print('PLEASE ENTER ONE OF THE FOLLOWING:')
        print(' \'load hearts\': loads image of hearts as forcing function')
        print(' \'heart\': analytical heart forcing function')
        print(' \'square\': square forcing function')
        print(' \'circle\': circle forcing function')
        
        flag = 0
    
    if flag:
        
        setup_img = BZ_solver( params_img )

        print('DISPLAYING THE FORCING FUNCTIONS:')

        setup_img.displForcing()

        print('PROPAGATING THE SOLVER:')

        setup_img.solver_RK4()

        print('GENERATING THE ANIMATION:')

        setup_img.generateAnimation( species=0, fName= sys.argv[1] + '.mp4')