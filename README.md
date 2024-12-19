# <center> Crawl Quest </center>

<p align="right" xmlns="http://www.w3.org/1999/html">
<a href="https://www.python.org" target="_blank" rel="noreferrer">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="20"/>
</a>
<img src="https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white" alt="Open CV"  height="20">

<img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy"  height="20">

<img src="https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white" alt="Ubuntu" height="20">

</p>

## Project Description

<div align="justify">

CrawlQuest is an AI-driven project designed to simulate and optimize a crawling agent's ability to navigate complex
environments. The agent learns to walk through a series of challenges using reinforcement learning algorithms,
specifically Q-Learning, Monte Carlo, and Temporal Difference (TD) SARSA. The goal of the project is to enable the agent
to adapt and improve its movement strategies through trial and error, allowing it to successfully crawl through various
terrain types.

The project leverages deep learning and reinforcement learning to enhance the agent's decision-making abilities, aiming
to create an intelligent, self-learning crawler that continuously refines its walking behavior. It combines theoretical
and practical elements of AI to provide insights into how machine learning models can be applied to real-world dynamic
environments. CrawlQuest serves as a unique exploration of training reinforcement learning agents for real-time
navigation tasks.

</div>

## Concepts

### **Monte Carlo Algorithm**

The Monte Carlo algorithm is a computational technique that relies on repeated random sampling to solve problems that
are deterministic in nature but computationally expensive or complex. It is widely used in optimization, numerical
integration, and probabilistic modeling.

---

#### **Key Concepts**

1. **Random Sampling**: Randomly generate inputs within the defined domain.
2. **Estimation**: Use the generated samples to compute approximate solutions for the problem.
3. **Convergence**: Accuracy improves as the number of samples increases.

---

#### **General Formula**

For a function \( f(x) \) over a domain \( D \):

\[
I $\approx$ $\frac{1}{N} \sum_{i=1}^N f(x_i)$
\]

Where:

- \( N \) = Number of samples.
- \( x_i \) = Random samples drawn uniformly from \( D \).

---

#### **Steps in Monte Carlo Simulation**

1. Define the domain \( D \) and the function \( f(x) \).
2. Generate \( N \) random points within \( D \).
3. Evaluate \( f(x_i) \) for each random sample \( x_i \).
4. Compute the average result using the formula above.

---

#### **Example: Estimating \( \pi \) Using Monte Carlo**

To estimate \( \pi \), consider a unit square with a quarter-circle of radius \( r = 1 \). Randomly generate points and
calculate the ratio of points inside the quarter-circle to total points.

\[
$\pi$ $\approx$ 4 $\times$ $\frac{\text{Number of points inside circle}}{\text{Total number of points}}$
\]

---

Monte Carlo algorithms are flexible and powerful, making them a cornerstone in machine learning, physics simulations,
and financial modeling.

### **Temporal Difference SARSA (TD-SARSA) Algorithm**

TD-SARSA (State-Action-Reward-State-Action) is an on-policy reinforcement learning algorithm that updates the
action-value function (Q-function) using the temporal difference (TD) method. It balances exploration and exploitation
by learning from the current policy.

---

#### **Key Concepts**

1. **On-Policy Learning**: SARSA learns the Q-values by following the policy being improved.
2. **Temporal Difference (TD) Update**: Combines ideas from Monte Carlo and Dynamic Programming for incremental updates.
3. **Bootstrapping**: Updates Q-values based on the current estimate of future rewards.

---

#### **Algorithm Overview**

For a given state \( s \), action \( a \), reward \( r \), next state \( s' \), and next action \( a' \), the Q-value
update rule is:

\[
Q(s, a) $\leftarrow$ Q(s, a) + $\alpha$ [ r + $\gamma$ Q(s', a') - Q(s, a) ]
\]

Where:

- \( $\alpha$ \): Learning rate (0 < \( $\alpha$ \) ≤ 1).
- \( $\gamma$ \): Discount factor (0 ≤ \( $\gamma$ \) ≤ 1).
- \( Q(s, a) \): Current action-value estimate.
- \( Q(s', a') \): Action-value estimate for the next state-action pair.

---

#### **Steps in TD-SARSA**

1. Initialize \( Q(s, a) \) for all states \( s \) and actions \( a \).
2. For each episode:
    - Start in an initial state \( s \) and select an action \( a \) using the policy \( \pi \) (e.g., \( \epsilon
      \)-greedy).
    - Repeat for each step:
        - Take action \( a \), observe reward \( r \), and the next state \( s' \).
        - Choose next action \( a' \) using policy \( \pi \).
        - Update \( Q(s, a) \) using the formula above.
        - Update \( s \leftarrow s' \), \( a \leftarrow a' \).
3. Update the policy \( \pi \) based on the new Q-values.

---

#### **Example**

Consider a grid-world environment:

1. Initialize \( Q(s, a) \) to zeros.
2. Use \( \epsilon \)-greedy policy for action selection.
3. Update \( Q(s, a) \) iteratively based on rewards and transitions.

---

TD-SARSA is effective in scenarios requiring real-time learning from ongoing interactions, making it suitable for tasks
like robotics, game AI, and dynamic decision-making systems.

### **Q-Learning Algorithm**

Q-Learning is an off-policy reinforcement learning algorithm that seeks to learn the optimal action-value function (
Q-function) by iteratively updating Q-values. It is widely used due to its simplicity and effectiveness in solving
Markov Decision Processes (MDPs).

---

#### **Key Concepts**

1. **Off-Policy Learning**: Q-Learning learns the optimal policy independently of the agent's current policy.
2. **Temporal Difference (TD) Learning**: Updates Q-values incrementally based on observed rewards and estimated future
   rewards.
3. **Exploration vs. Exploitation**: Uses strategies like \( \epsilon \)-greedy to balance exploration of the
   environment and exploitation of known rewards.

---

#### **Algorithm Overview**

For a given state \( s \), action \( a \), reward \( r \), and next state \( s' \), the Q-value update rule is:

\[
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
\]

Where:

- \( \alpha \): Learning rate (0 < \( \alpha \) ≤ 1).
- \( \gamma \): Discount factor (0 ≤ \( \gamma \) ≤ 1).
- \( Q(s, a) \): Current action-value estimate.
- \( \max_{a'} Q(s', a') \): Maximum estimated reward for the next state \( s' \).

---

#### **Steps in Q-Learning**

1. Initialize \( Q(s, a) \) for all states \( s \) and actions \( a \) (commonly initialized to zero).
2. For each episode:
    - Start in an initial state \( s \).
    - Repeat for each step:
        - Select an action \( a \) using an \( \epsilon \)-greedy policy.
        - Take action \( a \), observe reward \( r \) and next state \( s' \).
        - Update \( Q(s, a) \) using the formula above.
        - Update \( s \leftarrow s' \).
3. Continue until the episode ends or a terminal state is reached.

---

#### **Example**

Consider a grid-world environment:

1. Initialize \( Q(s, a) \) to zeros.
2. Use an \( \epsilon \)-greedy policy to select actions.
3. Update \( Q(s, a) \) iteratively based on rewards and transitions until the Q-values converge.

---

Q-Learning is effective for discrete action spaces and environments with deterministic or stochastic transitions. It is
foundational in tasks such as game playing, robotic control, and adaptive systems.

## Requirements

Operating System : Linux
<br>
Languages : Python 3.6

## Installation
Step 1: Setting Up a Virtual Environment

To create and activate a virtual environment, follow these steps:

* Run the following command to create a virtual environment named `venv`:


   ```bash
      python3 -m venv venv
   ```

Step 2: Activating the Virtual Environment

* Run the following command to activate a virtual environment named `venv`:


   ```bash
      source ./venv/bin/activate
   ```

Step 3: Installing requirements:

   ```bash
      pip install -r requirements.txt
   ```

Step 4: Executing the code

   ```bash
      python3 crawler_f.py
   ```


## Credits

<p>
1 . Ju Shen 

**Professor:** `University of Dayton`

 - jshen1@udayton.edu
</p>

## Contributions

# <div align="center"> Just Fork it and Implement you reinforcement algorithm </div>