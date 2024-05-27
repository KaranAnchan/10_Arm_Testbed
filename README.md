
---

# 10-Arm Testbed Simulation 🎰

## Overview 📖

This project implements a simulation of the 10-arm testbed problem commonly used in reinforcement learning to demonstrate the ε-greedy algorithm. Different ε-values are tested to observe their impact on the agent's ability to balance exploration and exploitation.

## Files in the Repository 🗂️

- `main.py`: The main script to run simulations. It sets up the environment, initializes agents with different ε-values, and runs the simulations.
- `agent.py`: Defines the `Agent` class, which encapsulates the behavior of an ε-greedy agent.
- `visualization.py`: Contains functions to visualize the results of the simulations using Seaborn and Matplotlib for better aesthetic appeal.

## Setup & Installation 🛠️

Before running the simulation, make sure you have Python installed on your system. You will also need the following Python packages:
- NumPy
- Matplotlib
- Seaborn

You can install these packages using pip:

```bash
pip install numpy matplotlib seaborn
```

## Running the Simulation 🚀
To run the simulation, execute the `main.py` file. This can be done from the command line:

```bash
python main.py
```

## Results 📊

### Average Reward vs. Episodes
![Average Reward vs. Episodes](images/average_reward_plot.png)
*This plot illustrates how the average reward received by the agents evolves over the episodes for different ε-values, including an agent with optimistic initial values. Observations from the visualization include:*
- **Optimistic Initial Values**: The agent with optimistic initial values tends to explore more aggressively in the early episodes. This can lead to higher variance in rewards initially but often stabilizes as the agent learns the true values of the arms.
- **Higher ε-values**: Agents with higher ε-values (e.g., 0.2) show more consistent exploration, which sometimes leads to discovering more rewarding options but can also result in lower average rewards due to frequent exploration.
- **Lower ε-values**: Agents with lower ε-values (e.g., 0.01) demonstrate more exploitation, quickly settling on what they initially perceive as the best option, leading to higher immediate returns but potential under-exploration.

### Selections of Each Arm
![Selections of Each Arm](images/selections_plot.png)
*This plot displays how frequently each arm was chosen by the agents, highlighting their exploration strategies. Key insights include:*
- **Balanced Exploration and Exploitation**: Agents with moderate ε-values (e.g., 0.1) often achieve a balance, adapting their strategy based on accumulated knowledge to focus increasingly on better-performing arms.
- **Optimistic Exploration**: The agent with optimistic initial values displays a more uniform exploration pattern initially, reducing the selection bias towards initially rewarding arms, which can lead to a more thorough understanding of the true value of all options.
- **Preference Patterns**: Variations in arm selections for higher ε-values suggest a strategy of widespread exploration, while lower ε-values show rapid convergence towards fewer arms, indicating strong exploitation.

## Contributing 🤝

Feel free to fork this project. Enjoy exploring reinforcement learning with this 10-arm testbed simulation! 🌟

## License 📄
This project is open-source and available under the MIT License.

---
