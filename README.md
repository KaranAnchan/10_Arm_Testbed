
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

The simulation generates two types of visualizations:

- **Average Reward vs. Episodes**: Shows how the average reward received by the agents evolves over the episodes.
  
- **Selections of Each Arm**: Displays how frequently each arm was chosen by the agents, indicating their exploration strategies.

## Contributing 🤝

Feel free to fork this project. Enjoy exploring reinforcement learning with this 10-arm testbed simulation! 🌟

## License 📄
This project is open-source and available under the MIT License.

---
