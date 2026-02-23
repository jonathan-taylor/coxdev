from itertools import combinations

import numpy as np
import pandas as pd

# basic model for times
rng = np.random.default_rng(0)
def sample_weights(size=1):
    W = rng.poisson(2, size=size) + rng.uniform(size=size)
    W[size//3] += 2.
    W[size//3:size//2] += 1
    return W

def sample_weights_zeros(size=1):
    W = rng.poisson(2, size=size) + rng.uniform(size=size)
    W[size//3] += 2.
    W[size//3:size//2] += 1
    W[rng.choice(size, size//5, replace=False)] = 0
    return W

def sample_times(size=1):
    W = rng.uniform(size=size) + 0.5
    W[size//3] += 2.
    W[size//3:size//2] += 1
    return W

# simulate different types of ties occurring
def simulate(start_count,
             event_count,
             size=1,
             rng=rng):

    size = rng.poisson(size) + 1
    #print(f'Size is {size}')
    if start_count == 0 and event_count == 0:
        return None

    # both of this well ensure there are unique starts or events 
    elif ((start_count == 0 and event_count == 1) or  
                                                      
          (start_count == 1 and event_count == 0)):
        start = sample_times(size=size)
        event = start + sample_times(size=size)
        
    # ties in event but not starts
    elif (start_count == 0) and (event_count == 2): 
        event = sample_times() * np.ones(size)
        start = event - sample_times(size=size)
        min_start = start.min()
        E = sample_times()
        event += min_start + E
        start += min_start + E
    # ties in starts but not events
    elif (start_count == 2) and (event_count == 0): 
        start = sample_times() * np.ones(size)
        event = start + sample_times(size=size)
    # single tie in start and event
    elif (start_count == 1) and (event_count == 1): 
        start = []
        event = []
        for _ in range(size):
            U = sample_times()
            start.extend([U-sample_times(), U])
            event.extend([U, U+sample_times()])
        start = np.asarray(start).reshape(-1)
        event = np.asarray(event).reshape(-1)
        
    # multiple starts at single event
    elif (start_count == 2) and (event_count == 1): 
        start = sample_times() * np.ones(size)
        event = start + sample_times(size=size)
        E = sample_times()
        event = np.hstack([event, start[0]])
        start = np.hstack([start, start[0] - sample_times()])

    # multiple events at single start
    elif (start_count == 1) and (event_count == 2):
        event = sample_times() * np.ones(size)
        start = event - sample_times(size=size)
        E = sample_times()
        start = np.hstack([start, event[0]])
        event = np.hstack([event, event[0] + sample_times()])

    # multiple events and starts
    elif (start_count == 2) and (event_count == 2): 
        U = sample_times()
        event = U * np.ones(size)
        start = event - sample_times(size=size)
        size2 = rng.poisson(size) + 1
        start2 = U * np.ones(size2)
        event2 = start2 + sample_times(size=size2)
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
                rng=rng,
                noinfo=True):
    dfs = []
    for tie_type in tie_types:
        for _ in range(nrep):
            dfs.append(simulate(tie_type[0],
                                tie_type[1],
                                size=size,
                                rng=rng))
    df = pd.concat(dfs)

    # if noinfo, include some points that have no failures
    # and are beyond the last failure time

    max_event = df['event'].max()
    if noinfo:
        start = max_event + rng.standard_exponential(5)
        event = start + rng.standard_exponential(5)
        df_noinfo = pd.DataFrame({'start':start,
                                  'event':event,
                                  'status':np.zeros(5)})
        df = pd.concat([df, df_noinfo])

    return df
        
