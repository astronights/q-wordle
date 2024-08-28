# Q Wordle

This project builds Reinforcement Learning Agents to learn how to play Wordle. Several agents are experimented with to identify th most optimal working solution with the best scores.

## Agents

The Reinforcement Learning Agents attempted are as follows:
- Random Action (Baseline)
- Q Learning
- SARSA
- Deep Q Learning
- A2C

## Gameplay

The gameplay for Wordle is replicated with the exact same set of rules as the original. There are 6 attemps to guess the word. For each guess, each letter of the word is flagged a certain colour.

- A correct letter in the correct position is marked as green
- A correct letter but in the wrong position (from the remaining letter) is marked as yellow
- Incorrect letters are marked as grey.

## Environment

A robust environment is setup after identifying the optimum action and state spaces, minimizing complexity and ensuring all functionalities are adequately captured.

### Reward

The reward function is calculated using a weighted score of the guess, with the values assigned to the 3 outcomes for each letter.

### State

The state space consists of a counter of the 3 observations for the given guess as well as the number of unexplored letters. This ensures the model generalizes to the puzzle by learning how to act based on the game feedback rather than the specific words it sees.

### Actions

The actions here are not the words to pick, but rather the strategies to use to pick words. The strategies here essentially replicate human behaviour to an extent to understand how the agent works. Some of the strategies used are as follows:

- Random Word
- Highest Log Likelihood Word
- Improve Guess
- Explore New Letters
