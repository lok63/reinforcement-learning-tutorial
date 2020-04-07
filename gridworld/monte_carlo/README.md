## Monte Carlo simulations for GridWorld

### 1.0 Solve the prediction Probelm
Given a policy we want to find the V(s)

#### 1.1 Fixed Policy
Things to know  :
* We need to have a pre-defined <b> policy </b>

`    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }`
    
| R | R | R |   |
|---|---|---|---|
| U |   | R |   |
| U | R | R | U |
 
* here the transition probabilities are deterministic, since we always have one choice to where to go to
* hence p(s',r| s,a) is 1 or 0
* Monte carlo here is not really necessary since the transitions are not pre-defined

### Example:
Episode 1:
 * We started randomly from position (0,0). Now we only have to move to the states according to our policy.
 * We only calculate the returns for those specific states
 * No need to calculate the returns for all states
 * [((0, 0), -0.7290000000000001), ((2, 1), -0.81), ((2, 2), -0.9), ((2, 3), -1.0), ((1, 3), 0)]
 
 
 
 ### 1.2 Windy GridWord (Random Policy)
Things to know  :
* We have a different policy than the first example 

| R | R | R |   |
|---|---|---|---|
| U |   | U |   |
| U | L | U | L |


* We have a source of randomnsess here because we don't pre-define the transitions
* That's where Monte Carlo kicks in

### 2.0 Find the optimal policy
* We only have V(s) for a given policy
* We don't know what actions will lead to better V(s) because we can't look-ahead search



