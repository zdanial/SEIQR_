# SEIQR_: Python simulation code for "Three pre-vaccine responses to Covid-like epidemics"
### Lai-Sang Young, Zach Danial


To get started: 
```
import numpy as np
from SEIQR import SEIQR
```

Then create initial values:
```
I0 = 10e-5
S0, E0, R0=0, 0, 0  # S0 = 0 is handled within the SEIQR class

beta_0 = 2.5
m_0 = 1
r0=(2/3)*beta_0*3*m_0

t0 = int((0.35+0.25+1.5)/0.005)-1

t_vec = np.arange(0, 100, 0.005)

r1 = beta*m
p=0.2
```

Pass into a dictionary 
```
base_prams=dict(
  beta= (2/3)*5/2, #r1
  m= 3*m_0,
  alpha= 0.005,
  gamma= 1,
  sigma= 0.35,#0.3,
  tau= 0.25,#0.2,
  kappa=1.5,
  rand='e',

  epsilon= lambda t, qp: 1,
  p= 0,
  
  delta=1,

  t_start=0, 
  dt=0.005,
  t_end=100,

  init=(S0, E0, I0, R0)
)

```

Create an instance and run
```
x=SEIQR(**base_prams)
x.run()
```

Then you can plot all the different system components.