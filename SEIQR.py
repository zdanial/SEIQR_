import numpy as np
from tqdm import tqdm
import pandas as pd

class SEIQR():

  def __init__(self, **kwargs):

    """
    Initialization of SEIQR simulation object.  Pass a dictionary with the following key, value pairs for the **kwargs:
    simga: scalar
    tau: scalar
    kappa: scalar
    epsilon: scalar
    alpha: scalar
    t_start: 0 (for default)
    dt: 0.005 (for default)

    r1: vector; or (beta: vector and m: scalar)


    qpld: scalar
    dld: scalar
    delta_o: scalar
    qpo: scalar

    """
    # Parameters are passed through kwargs, missing required keys cause an error through 'assert'
    keys=kwargs.keys()
    self.kwargs = kwargs

    assert 'sigma' in keys
    self.sigma=kwargs['sigma']
    assert 'tau' in keys
    self.tau=kwargs['tau']
    assert 'kappa' in keys
    self.kappa=kwargs['kappa']
    assert 'epsilon' in keys
    self.epsilon=kwargs['epsilon']
    if 'alpha' in keys:
      self.alpha=kwargs['alpha']
    else:
      self.alpha=0
    
    # Creation of time vector
    assert 't_start' in keys
    assert 'dt' in keys
    self.t_start=kwargs['t_start']
    if 't_end' in keys:
      self.t_end=kwargs['t_end']
    else:
      self.t_end=80
    self.dt=kwargs['dt']
    self.t_vector = np.arange(self.t_start, self.t_end, self.dt) # vector for each point in time
    self.t0=int((self.sigma+self.tau+self.kappa)/self.dt)-1
    self.size=self.t_vector.shape[0]


    # first lockdown logic
    self.lockdown_time = None # this is used to test if first lockdown has occured yet
    if 'qpld' in keys: # qpld = Q+_lockdown; the daily cases threshold for first lockdown
      self.qpld=kwargs['qpld']
    if 'dld' in keys: # dld = delta_lockdown; the value to set delta when Q+ > qpld
      self.dld=kwargs['dld']
    if 'steer_value' in keys:
      self.steer_value=kwargs['steer_value']
    if 'delta_o' in keys: # delta_o = delta_open; the value to set delta when Q+ < qpo
      self.delta_o=kwargs['delta_o']
    if 'qpo' in keys: # qpo = Q+_open; the daily cases threshold for re-opening
      self.qpo=kwargs['qpo']
    
      
    # if 'r1' in keys:
    #   self.r1 = kwargs['r1'] # sets external value (or vector) for reproductive number
    if 'beta' in keys:
      self.beta=np.zeros(self.size) + kwargs['beta'] # creation of beta(t) data-structure
    assert 'm' in keys
    self.m=kwargs['m']
    
    self.stop_spikes=False


    # Set Initial Conditions
    assert 'init' in keys
    self.X0=kwargs['init']
    
    self.S=np.zeros(self.size)
    
    self.E=np.zeros(self.size)
    
    self.I=np.zeros(self.size)
    
    self.Q=np.zeros(self.size)
    self.Qp=np.zeros(self.size)

    self.Qpb=np.zeros(self.size)
    self.Ep=np.zeros(self.size)

    self.R=np.zeros(self.size)
    self.rset = False


    if self.t0>0:
      I0=self.X0[2]
      self.I[:self.t0+1]=[I0*np.exp(self.t_vector[i] - self.t_vector[self.t0]) for i in range(self.t0+1)]
      self.S[:self.t0+1]=1-self.I[:self.t0+1]-self.E[:self.t0+1]-self.Q[:self.t0+1]-self.R[:self.t0+1]


    elif self.t0==0:
      self.S[0]=self.X0[0]
      self.E[0]=self.X0[1]
      self.I[0]=self.X0[2]
      self.Q[0]=0
      self.Qp[0]=0
      self.R[0]=self.X0[3]
    

    if 'd_func' in keys:
      self.delta=np.ones(self.size)
      self.d_func=kwargs['d_func']
      self.has_d_func=True
      if 'd_param' in keys:
        self.d_param=kwargs['d_param']
      else:
        self.d_param=None
    elif 'delta' in keys:
      self.delta=np.zeros(self.size) + kwargs['delta']
      self.has_d_func=False
    self.epsilon_history = np.ones(self.size)
    self.n_history = np.zeros(self.size)
    self.n_historyb = np.zeros(self.size)
    # if kwargs['rand']=='manual':  # to manually test stochastic component uncomment this code
    #   self.n_history[int(24/self.dt)] = 1000/10e6
    self.ct_history = np.zeros(self.size)

    if 'p_func' in keys:
      self.p=np.zeros(self.size)
      self.p_func=kwargs['p_func']
      self.has_p_func=True
      if 'p_param' in keys:
        self.p_param=kwargs['p_param']
      else:
        self.p_param=None
    elif 'p' in keys:
      self.p=np.zeros(self.size) + kwargs['p']
      self.has_p_func=False










  def run(self):
    """
    Main loop to run simulation from t_start to t_end
    """
    # Booleans for multi-stage lockdown strategy
    first = False
    last = False
    mid = False
    done= False


    for i, t in enumerate(self.t_vector[:-1]):
      if i>=self.t0:

        # This code-block sets delta according to a defined delta function.  For multi-stage strategies this
        # requires access to flipping the booleans defined at the top of this function.  This needs to be switched
        # to attributes on the class itself so it can be moved to a separate function.
        if self.has_d_func:
          if self.d_param=='Q+':
            self.delta[i]=self.d_func(i, self.R, self.Qp, self.Q)
          elif self.d_param=='straight':
            self.delta[i]=SEIQR.straight(self, i)

          elif self.d_param=='new_model3BIG':
            delayed_idx = int(i-0.5/self.dt)
            r_eff = self.beta[i]*self.m*(1-self.p[i]*np.exp(-self.tau))
            d=self.d_func/(r_eff*(1-self.R[delayed_idx]-self.Q[delayed_idx]))
            if (self.Qp[i]>0.003 or done) and mid and last:
              self.delta[i]=np.min([d,1])
              done=True
            elif first and ((self.Qp[i]<0.001 and t>14) or mid):
              if not mid:
                self.delta[i:int(i+7.5/self.dt)]= np.min([d,1])
                self.delta[int(i+7.5/self.dt):]=0
              elif mid:
                if self.delta[i]==0:
                  self.delta[i] = np.min([1.25*d,1])
              mid = True
              last = False if self.Qp[i]<0.003 else True
            elif self.Qp[i]>0.002 and not first:
              self.delta[i:]=self.dld
              first = True
            elif not first and not last: 
              self.delta[i] = 1
          
          elif self.d_param=='delta_open':
            if first and self.Qp[i]<self.qpo and not last:
              self.delta[i:]= self.delta_o
              last = True
            elif self.Qp[i]>0.002 and not first:
              self.delta[i:]=self.dld
              first = True
            elif not first and not last: 
              self.delta[i] = 1

          elif self.d_param=='periodic':
            if last and self.Qp[i]>self.qpld and self.delta[i]!=0:
              self.delta[i:i+400]=self.dld
              self.delta[i+400:]=0
              last = False
            elif first and (last or self.Qp[i]<self.qpo or self.delta[i]==0):
              delayed_idx = int(i-0.5/self.dt)
              r_eff = self.beta[delayed_idx]*self.m*(1-self.p[delayed_idx]*np.exp(-self.tau))
              d=self.d_func/(r_eff*(1-self.R[delayed_idx]-self.Q[delayed_idx]))
              self.delta[i]= np.min([d, 1])
              self.delta[i+1:]= 1
              last = True
            elif self.Qp[i]>0.002 and not first:
              self.delta[i:]=self.dld
              first = True
            elif not first and not last: 
              self.delta[i] = 1

          elif self.d_param=='new_periodic':
            delayed_idx = int(i-0.5/self.dt)
            r_eff = self.beta[delayed_idx]*self.m*(1-self.p[delayed_idx]*np.exp(-self.tau))
            d=self.d_func/(r_eff*(1-self.R[delayed_idx]-self.Q[delayed_idx]))
            d1=1/(r_eff*(1-self.R[delayed_idx]-self.Q[delayed_idx]))
            d2=1.09/(r_eff*(1-self.R[delayed_idx]-self.Q[delayed_idx]))
            if last and self.Qp[i]>self.qpld:
              self.delta[i:i+400]=self.dld
              self.delta[i+400:]=1
              last = False
            elif first and (last or self.Qp[i]<self.qpo or self.delta[i]==1):
              if not mid:
                self.delta[i:int(i+7.5/self.dt)]= np.min([d1,1])
                self.delta[int(i+7.5/self.dt):]=0
              elif mid:
                if self.delta[i]==0:
                  self.delta[i]= np.min([d, 1])
                elif self.delta[i]==1:
                  self.delta[i]= np.min([d2, 1])
              mid=True
              last = True
            elif self.Qp[i]>0.002 and not first:
              self.delta[i:]=self.dld
              first = True
            elif not first and not last:
              self.delta[i] = 1
            
            
        self.set_nhistory(i, t)
        self.step(i, t)



  def step(self, i, t):
    """
    Function to updaate the SEIQR object each step of time
    """
    # Array indexes for different delays
    # d=int(i-self.sigma/self.dt)
    # ta=int(i-(self.sigma+self.tau)/self.dt)
    # tt = int(i-self.tau/self.dt)

    # ta2=int(i-(self.sigma+2*self.tau)/self.dt)
    # tt2 = int(i-2*self.tau/self.dt)
    # ta3=int(i-(self.sigma+3*self.tau)/self.dt)
    # tt3 = int(i-3*self.tau/self.dt)

    # tk = int(i-(self.tau+self.kappa)/self.dt)
    # tka=int(i-(self.sigma+self.tau + self.kappa)/self.dt)
    # tka2=int(i-(self.sigma+2*self.tau+self.kappa)/self.dt)
    # tk2 = int(i-(2*self.tau+self.kappa)/self.dt)
    # tka3=int(i-(self.sigma+3*self.tau+self.kappa)/self.dt)
    # tk3 = int(i-(3*self.tau+self.kappa)/self.dt)

    # Update the array for each variable
    self.epsilon_history[i] = self.epsilon(t, self.Qpb[int(i - self.tau/self.dt)])
    self.S[i+1]=self.S[i] + self.sdot(i, t)*self.dt
    self.E[i+1]=self.E[i] + self.edot(i, t)*self.dt
    self.I[i+1]=self.I[i] + self.idot(i, t)*self.dt 
    self.Q[i+1]=self.Q[i] + self.qdot(i, t)*self.dt
    self.Qp[i+1]+=self.qplus(i, t)
    self.Qpb[i+1]+=self.qplus_base(i, t)
    self.Ep[i+1] = self.delta[i]*self.beta[i]*self.m*self.S[i]*self.I[i] + self.n_history[i]/self.dt
    self.R[i+1]=self.R[i] + self.rdot(i, t)*self.dt
    

  # Derivative Formulae
  def sdot(self, i, t):
    return -self.Ep[i] + self.alpha*self.R[i]

  def edot(self, i, t):
    d=int(i-self.sigma/self.dt)
    return self.Ep[i] - self.Ep[d]

  def idot(self, i, t):
    d=int(i-self.sigma/self.dt)
    t=int(i-(self.sigma+self.tau)/self.dt)
    # pt=int(i-(self.tau)/self.dt)
    return self.epsilon_history[d]*(self.Ep[d] - self.n_history[d-1]/self.dt) - self.I[i] - self.Qp[i] + self.n_history[d-1]/self.dt
 
  def qdot(self, i, t):
    d=int(i-self.sigma/self.dt)
    t=int(i- (self.sigma+self.tau)/self.dt)
    # pt = int(i- self.tau/self.dt)
    # ptk = int(i- (self.tau+self.kappa)/self.dt)
    # k=int(i - (self.sigma+self.tau+self.kappa)/self.dt)
    k_prime=int(i - (self.sigma+self.kappa)/self.dt)
    ct = (1-self.epsilon_history[d])*(self.Ep[d] - self.n_history[d-1]/self.dt) - (1-self.epsilon_history[k_prime])*(self.Ep[k_prime] - self.n_history[k_prime-1]/self.dt)
    self.ct_history[i] = (1-self.epsilon_history[d])*(self.Ep[d] - self.n_history[d-1]/self.dt)
    return self.Qp[i] - self.Qp[int(i-self.kappa/self.dt)] + ct
  
  def qplus(self, i, t):
    t=int(i-(self.sigma+self.tau)/self.dt)
    t2=int(i-(self.sigma+2*self.tau)/self.dt)
    # t3=int(i-(self.sigma+3*self.tau)/self.dt)
    pt = int(i-self.tau/self.dt)
    pt2=int(i-(2*self.tau)/self.dt)
    # pt3=int(i-(3*self.tau)/self.dt)
    # k=int(i-(self.sigma+self.tau+self.kappa)/self.dt)
    return self.epsilon_history[t]*self.p[pt]*np.exp(-self.tau)*(self.Ep[t]- self.n_history[t-1]/self.dt) + self.p[pt]*np.exp(-self.tau)*self.n_history[t-1]/self.dt + (1/2)*(1 - self.p[pt2])*np.exp(-2*self.tau)*self.n_history[t2-1]/self.dt

  def qplus_base(self, i, t):   
    t=int(i-(self.sigma+self.tau)/self.dt)
    # t2=int(i-(self.sigma+2*self.tau)/self.dt)
    # t3=int(i-(self.sigma+3*self.tau)/self.dt)
    pt = int(i-self.tau/self.dt)
    # pt2=int(i-(2*self.tau)/self.dt)
    # pt3=int(i-(3*self.tau)/self.dt)
    # k=int(i-(self.sigma+self.tau+self.kappa)/self.dt)
    return self.epsilon_history[t]*self.p[pt]*np.exp(-self.tau)*(self.Ep[t]- self.n_history[t-1]/self.dt)

  def rdot(self, i, t):
    # d=int(i-self.sigma/self.dt)
    t=int(i-(self.sigma+self.tau)/self.dt)
    k=int(i-(self.sigma+self.tau+self.kappa)/self.dt)
    k_prime=int(i - (self.sigma+self.kappa)/self.dt)
    # ptk = int(i- (self.tau+self.kappa)/self.dt)
    ct = (1-self.epsilon_history[k_prime])*(self.Ep[k_prime] - self.n_history[k_prime-1]/self.dt)
    return -self.alpha*self.R[i] + self.I[i] + self.Qp[k] + ct



  # Helper functions
  def tld(self, t=-1):
    try:
      w=np.where(self.delta[:t+1] < 1)[0][0]
    except IndexError:
      return None
    return self.t_vector[w], w

  def set_nhistory(self, i, t):
    """
    Handles stochastic compnent.  b, c, d, e are some presets for random superspreading events.
    """
    if t%2==0 and not self.stop_spikes:
      s = self.kwargs['rand']
      r =  np.random.uniform()
      if s =='b':
        if r<= 0.1:
          n = 20
          self.n_history[i] += n/10e6
      elif s =='c':
        if r<= 0.1:
          n = np.random.randint(30, 50)
          self.n_history[i] += n/10e6
      elif s =='d':
        if r<= 0.1:
          n = np.random.randint(1, 100)
          self.n_history[i] += n/10e6
      elif s =='e':
        if r<= 0.1:
          n = np.random.randint(1, 280)
          self.n_history[i] += n/10e6
      else:
        pass

  @staticmethod
  def transition(start_val, end_val, start_time, end_time, t_vec):
    out=np.zeros(t_vec.shape[0]) + end_val
    t0=np.where(t_vec>start_time)[0][0]
    t1=np.where(t_vec>end_time)[0][0]
    out[:t0]=start_val
    rng=t1-t0
    out[t0:t1]=[start_val + (end_val-start_val)*i**2/rng**2 for i in range(rng)]
    return out

  @staticmethod
  def straight(x, i, delay=1/2, dld=0.3, qpld=0.002):
    ld = False if not i > 1 else np.max(x.Qp[:i]) > qpld
    ld_end = False
    if ld and not x.kwargs['sett']:
      x.lockdown_time = i
      print('set')
      x.kwargs['sett'] = True


    if ld and i - x.lockdown_time > 1:
      try:
        x.max_Q
      except:
        x.max_Q=np.argmax(x.Q)
        
      if i-x.max_Q>1:
        ld_end = np.min(x.Qp[x.max_Q:i])<x.steer_value
      else:
        ld_end = False

    
    if ld_end and ld:
      if x.Qp[i]>=x.steer_value and i*x.dt<30:
        return x.delta[i-1]

      delayed_idx = int(i-0.5/x.dt)
      r_eff = x.beta[delayed_idx]*x.m*(1-x.p[delayed_idx]*np.exp(-x.tau))
      d=x.d_func/(r_eff*(1-x.R[delayed_idx]-x.Q[delayed_idx]))
      return np.min([d, 1])
    if ld:
      return dld
    else:
      return 1


  @staticmethod
  def contact_curve(n):
    daily = n*10000000/10
    return 1/(1+np.exp((daily-100)/20))

  @staticmethod
  def delta_open(x, i, r_eff=False):
    # iopen = np.where(x.delta[1:]>x.delta[:-1])[0][0]
    delay = int(i - x.tau/x.dt)
    r1 = np.multiply(x.beta, x.m)
    r2 = r1[delay]*(1 - x.p[delay]*np.exp(-x.tau))
    r3 = r2*(1 - x.R[delay])
    r4 = r3*x.epsilon(x.t_vector[delay], x.Qp[delay])
    if r_eff:
      return r4
    return 1/r4

  @staticmethod
  def test(x, i):
    delay = i#int(i - x.tau/x.dt)
    r1 = np.multiply(x.beta, x.m)
    r2 = r1[delay]*(1 - x.p[delay]*np.exp(-x.tau))
    r3 = r2*(1 - x.R[delay] - x.Q[delay])
    r4 = r3*x.delta_o
    return (1/r4)

     
