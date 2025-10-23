# Test-Time Scaling for TextArena

This guide explains how to implement and evaluate test-time scaling approaches on TextArena benchmarks.

## Overview

**Test-time scaling** refers to techniques that use additional computation at inference time to improve model performance. Instead of just generating a single response, these methods:

- Generate multiple candidates and select the best
- Iteratively refine responses
- Use chain-of-thought reasoning with verification
- Employ search or planning algorithms

## Quick Start

### 1. Install Dependencies

```bash
pip install textarena
# Set your API key
export OPENROUTER_API_KEY="your-key-here"
```

### 2. Run a Simple Test

```python
import textarena as ta
from test_time_scaling_examples import SelfConsistencyWrapper

# Create base agent
base_agent = ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku")

# Wrap with test-time scaling
agent = SelfConsistencyWrapper(base_agent, n_samples=5)

# Use in a game
agents = {0: agent, 1: base_agent}
env = ta.make("TicTacToe-v0")
env.reset(num_players=2)

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agents[player_id](observation)
    done, step_info = env.step(action=action)

rewards, game_info = env.close()
print(f"Rewards: {rewards}")
```

### 3. Run Full Evaluation

```bash
# Evaluate on TicTacToe
python evaluate_scaling_approaches.py --env TicTacToe-v0 --n-games 10

# Evaluate on multiple games
python evaluate_scaling_approaches.py \
    --env TicTacToe-v0 ConnectFour-v0 Othello-v0 \
    --n-games 20 \
    --approaches self_consistency_5 best_of_3 baseline

# Use different models
python evaluate_scaling_approaches.py \
    --env Chess-v0 \
    --model gpt-4o-mini \
    --n-games 5
```

## Available Test-Time Scaling Approaches

### 1. **Self-Consistency** (Best for discrete action spaces)

Generates multiple independent samples and selects the most common action via majority vote.

**Use when:**
- Actions are discrete (board positions, chess moves)
- Correct answer is likely to be consistent across samples

**Scaling factor:** Number of samples (typical: 3-10)

```python
from test_time_scaling_examples import SelfConsistencyWrapper

agent = SelfConsistencyWrapper(
    base_agent,
    n_samples=5,          # Generate 5 samples
    temperature=0.7,      # Sampling temperature
    debugging=True        # Print samples and vote distribution
)
```

**Computational cost:** O(n_samples)

---

### 2. **Best-of-N Sampling** (Best for strategic decisions)

Generates N candidate actions, then uses the model to evaluate and select the best one.

**Use when:**
- Need quality over diversity
- Strategic games where evaluation is easier than generation

**Scaling factor:** Number of candidates (typical: 3-5)

```python
from test_time_scaling_examples import BestOfNWrapper

agent = BestOfNWrapper(
    base_agent,
    n_candidates=3,       # Generate 3 candidates
    evaluation_prompt="Select the action that maximizes winning probability",
    debugging=True
)
```

**Computational cost:** O(n_candidates + 1) for generation + evaluation

---

### 3. **Iterative Refinement** (Best for complex decisions)

Generates an initial action, critiques it, then refines based on the critique. Can iterate multiple times.

**Use when:**
- Complex strategic situations
- Initial responses tend to be suboptimal
- Self-critique can identify weaknesses

**Scaling factor:** Number of iterations (typical: 1-3)

```python
from test_time_scaling_examples import IterativeRefinementWrapper

agent = IterativeRefinementWrapper(
    base_agent,
    n_iterations=2,       # Refine twice
    critique_prompt="What are the strengths and weaknesses of this action?",
    refinement_prompt="Improve the action based on the critique",
    debugging=True
)
```

**Computational cost:** O(2 × n_iterations + 1)

---

### 4. **Chain-of-Thought with Verification** (Best for logical reasoning)

Generates step-by-step reasoning, then verifies the final answer makes sense.

**Use when:**
- Games require logical reasoning
- Can verify correctness more easily than generate
- Reasoning transparency is valuable

**Scaling factor:** Verification attempts (typical: 1-2)

```python
from test_time_scaling_examples import CoTVerificationWrapper

agent = CoTVerificationWrapper(
    base_agent,
    max_verification_attempts=2,
    debugging=True
)
```

**Computational cost:** O(1 + verification_attempts)

---

### 5. **Ensemble Strategies** (Best for complex games)

Uses multiple different prompting strategies and combines their outputs.

**Use when:**
- Different approaches have different strengths
- Game has multiple valid strategies

**Scaling factor:** Number of strategies (typical: 3-5)

```python
from test_time_scaling_examples import EnsembleStrategyWrapper

agent = EnsembleStrategyWrapper(
    base_agent,
    strategies=[
        "Play aggressively to maximize immediate advantage.",
        "Play defensively to minimize opponent's options.",
        "Play strategically for long-term position.",
    ],
    combination_method="vote",  # or "evaluate"
    debugging=True
)
```

**Computational cost:** O(n_strategies) or O(n_strategies + 1) for evaluation

---

### 6. **Simplified MCTS** (Best for turn-based games)

Simulates multiple action sequences and evaluates outcomes to select the best first move.

**Use when:**
- Turn-based games with clear win conditions
- Can imagine/simulate future states
- Planning depth is important

**Scaling factor:** Rollouts × depth (typical: 3 rollouts × 3 depth)

```python
from test_time_scaling_examples import SimplifiedMCTSWrapper

agent = SimplifiedMCTSWrapper(
    base_agent,
    n_rollouts=3,         # Number of rollout simulations
    rollout_depth=3,      # How many moves ahead to simulate
    debugging=True
)
```

**Computational cost:** O(n_rollouts × rollout_depth)

**Note:** This is a simplified version using imagination, not true MCTS with environment simulation.

---

### 7. **Temperature Ladder** (Best for exploration-exploitation balance)

Samples at different temperature settings to balance creativity and consistency.

**Use when:**
- Want to explore multiple action types
- Balance between greedy and creative play

**Scaling factor:** Number of temperature levels (typical: 3-5)

```python
from test_time_scaling_examples import TemperatureLadderWrapper

agent = TemperatureLadderWrapper(
    base_agent,
    temperatures=[0.3, 0.7, 1.0],
    selection_method="vote",  # "vote", "median", or "highest_temp"
    debugging=True
)
```

**Computational cost:** O(n_temperatures)

---

## Combining Multiple Approaches

You can chain wrappers for even more powerful scaling:

```python
# Self-consistency with iterative refinement
agent = SelfConsistencyWrapper(
    IterativeRefinementWrapper(base_agent, n_iterations=1),
    n_samples=3
)

# CoT with verification + best-of-N
refined_agent = IterativeRefinementWrapper(base_agent, n_iterations=1)
agent = BestOfNWrapper(
    CoTVerificationWrapper(refined_agent),
    n_candidates=3
)
```

**Warning:** Chaining increases computational cost multiplicatively!

---

## Evaluation Results Format

The evaluation script saves results in JSON format:

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "config": {
    "base_agent_type": "openrouter",
    "model_name": "anthropic/claude-3.5-haiku",
    "n_games_per_env": 10
  },
  "evaluations": [
    {
      "approach": "self_consistency_5",
      "env_id": "TicTacToe-v0",
      "n_games": 10,
      "wins": 7,
      "draws": 2,
      "losses": 1,
      "win_rate": 0.7,
      "avg_time_seconds": 12.3,
      "avg_turns": 8.5,
      "results": [...]
    }
  ]
}
```

---

## Recommended Game Categories for Testing

### Fast Games (Good for initial testing)
- `TicTacToe-v0` - Classic 3x3
- `ReverseTicTacToe-v0` - Reverse objective
- `ConnectFour-v0` - 6x7 grid
- `Chopsticks-v0` - Hand game

### Strategic Games (Test reasoning)
- `Chess-v0` - Full chess
- `Othello-v0` - Reversi
- `Checkers-v0` - Draughts
- `Breakthrough-v0` - Pawn breakthrough

### Theory of Mind Games (Test social reasoning)
- `SecretMafia-v0` - Deduction game
- `Debate-v0` - Argumentation
- `CharacterConclave-v0` - Role-playing
- `TwentyQuestions-v0` - Q&A game

### Puzzle Games (Single player)
- `Minesweeper-v0` - Mine sweeping
- `TowerOfHanoi-v0` - Tower puzzle
- `FifteenPuzzle-v0` - Sliding puzzle
- `Sudoku-v0` - Number placement

### Negotiation Games (Test cooperation)
- `SimpleNegotiation-v0` - Basic bargaining
- `UsedCarNegotiation-v0` - Car sale
- `Diplomacy-v0` - Alliance building

---

## Analyzing Results

After running evaluations, analyze the results:

```python
import json

# Load results
with open("scaling_results/evaluation_20250115_103000.json") as f:
    results = json.load(f)

# Compare win rates
for eval in results["evaluations"]:
    print(f"{eval['approach']:30s} {eval['win_rate']:.1%}")

# Find best approach per environment
envs = {}
for eval in results["evaluations"]:
    env_id = eval["env_id"]
    if env_id not in envs:
        envs[env_id] = []
    envs[env_id].append(eval)

for env_id, evals in envs.items():
    best = max(evals, key=lambda x: x["win_rate"])
    print(f"{env_id}: {best['approach']} ({best['win_rate']:.1%})")
```

---

## Cost-Performance Tradeoffs

| Approach | Compute Cost | Typical Improvement | Best For |
|----------|--------------|---------------------|----------|
| Baseline | 1x | - | - |
| Self-Consistency (n=5) | 5x | +10-20% | Discrete actions |
| Best-of-N (n=3) | 4x | +5-15% | Strategic games |
| Iterative Refinement (n=2) | 5x | +10-25% | Complex decisions |
| CoT Verification | 2-3x | +5-10% | Logical reasoning |
| Ensemble (n=3) | 3-4x | +5-15% | Multi-strategy games |
| MCTS (3x3) | 9x | +15-30% | Planning games |
| Temperature Ladder (n=3) | 3x | +5-10% | Exploration |

**Note:** Actual improvements depend heavily on:
- Base model capability
- Game complexity
- Quality of prompts
- Opponent strength

---

## Creating Custom Scaling Approaches

To create your own scaling wrapper:

```python
import textarena as ta

class CustomScalingWrapper(ta.AgentWrapper):
    def __init__(self, agent: ta.Agent, your_param: int, debugging: bool = False):
        super().__init__(agent)
        self.your_param = your_param
        self.debugging = debugging

    def __call__(self, observation: str) -> str:
        """
        Your scaling logic here.

        You can:
        - Call self.agent(observation) multiple times
        - Modify self.agent.system_prompt temporarily
        - Access self.agent.kwargs for temperature, etc.
        """
        original_prompt = self.agent.system_prompt

        # Your implementation
        # ...

        # Always restore original state!
        self.agent.system_prompt = original_prompt

        return final_action
```

**Key patterns:**
1. Always save and restore `agent.system_prompt`
2. Use `debugging` flag for verbose output
3. Handle errors gracefully
4. Consider API rate limits for multiple calls

---

## Tips for Research

### 1. Start Small
- Test on fast games first (TicTacToe, Chopsticks)
- Use small n_games (5-10) for initial experiments
- Only scale to larger evaluations after validating approach

### 2. Control Variables
- Use fixed seeds for reproducibility
- Compare against same baseline
- Alternate starting positions (player 0 vs player 1)

### 3. Track Costs
- Monitor API costs (especially for high-sample approaches)
- Consider using cheaper models for experimentation
- Use temperature=0 for baseline comparisons

### 4. Analyze Failures
- Look at individual game logs
- Check if scaling helps on specific game situations
- Identify when approaches fail

### 5. Combine Approaches Strategically
- Chain complementary approaches
- Don't stack similar approaches (e.g., multiple self-consistency layers)
- Consider diminishing returns

---

## Common Issues

### Issue: "API rate limit exceeded"
**Solution:** Add delays between games or reduce n_samples

```python
import time
time.sleep(1)  # Add between games
```

### Issue: "Wrapper increases cost but not performance"
**Solution:** Try different parameters or check if base model is already strong

### Issue: "Inconsistent results"
**Solution:** Increase n_games for better statistics, use fixed seeds

### Issue: "Games take too long"
**Solution:** Start with faster games, reduce scaling parameters, or use cheaper models

---

## Further Reading

- TextArena paper: https://arxiv.org/abs/2504.11442
- Self-consistency: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
- Best-of-N: "Training Verifiers to Solve Math Word Problems"
- MCTS for LLMs: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
- Iterative refinement: "Self-Refine: Iterative Refinement with Self-Feedback"

---

## Contributing

Have a new test-time scaling approach? Add it to `test_time_scaling_examples.py`!

1. Inherit from `ta.AgentWrapper`
2. Implement `__call__(self, observation: str) -> str`
3. Add docstring with use cases and scaling factors
4. Test on at least 3 different game types
5. Document computational cost

---

## License

MIT License - See TextArena main repository for details.
