from itertools import combinations

import numpy as np
import pandas as pd

# basic model for times
rng = np.random.default_rng(0)
def sample(size=1):
    W = rng.uniform(size=size) + 0.5
    W[size//3] += 2
    W[size//3:size//2] += 1
    return W

# simulate different types of ties occurring
def simulate(start_count,
             event_count,
             size=1,
             rng=rng):

    size = rng.poisson(size) + 1
    if start_count == 0 and event_count == 0:
        return None

    # both of this well ensure there are unique starts or events 
    elif ((start_count == 0 and event_count == 1) or  
                                                      
          (start_count == 1 and event_count == 0)):
        start = sample(size=size)
        event = start + sample(size=size)
        
    # ties in event but not starts
    elif (start_count == 0) and (event_count == 2): 
        event = sample() * np.ones(size)
        start = event - sample(size=size)
        min_start = start.min()
        E = sample()
        event += min_start + E
        start += min_start + E
    # ties in starts but not events
    elif (start_count == 2) and (event_count == 0): 
        start = sample() * np.ones(size)
        event = start + sample(size=size)
    # single tie in start and event
    elif (start_count == 1) and (event_count == 1): 
        start = []
        event = []
        for _ in range(size):
            U = sample()
            start.extend([U-sample(), U])
            event.extend([U, U+sample()])
        start = np.asarray(start).reshape(-1)
        event = np.asarray(event).reshape(-1)
        
    # multiple starts at single event
    elif (start_count == 2) and (event_count == 1): 
        start = sample() * np.ones(size)
        event = start + sample(size=size)
        E = sample()
        event = np.hstack([event, start[0]])
        start = np.hstack([start, start[0] - sample()])

    # multiple events at single start
    elif (start_count == 1) and (event_count == 2):
        event = sample() * np.ones(size)
        start = event - sample(size=size)
        E = sample()
        start = np.hstack([start, event[0]])
        event = np.hstack([event, event[0] + sample()])

    # multiple events and starts
    elif (start_count == 2) and (event_count == 2): 
        U = sample()
        event = U * np.ones(size)
        start = event - sample(size=size)
        size2 = rng.poisson(size) + 1
        start2 = U * np.ones(size2)
        event2 = start2 + sample(size=size2)
        start = np.hstack([start, start2])
        event = np.hstack([event, event2])

    size = start.shape[0]
    status = rng.choice([0,1], size=size)
    return pd.DataFrame({'start':start, 'event':event, 'status':status})

dataset_types = [(0,1), (1,0), (1, 1), (0, 2), (2, 0), (2, 1), (1, 2), (2, 2)]
all_combos = []
for i in range(1, 9):
    for v in combinations(dataset_types, i):
        all_combos.append(v)

def simulate_df(tie_types,
                nrep,
                size,
                rng=rng):
    dfs = []
    for tie_type in tie_types:
        for _ in range(nrep):
            dfs.append(simulate(tie_type[0],
                                tie_type[1],
                                size=size,
                                rng=rng))
    return pd.concat(dfs)
        
