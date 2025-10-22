"""
Demo test-time scaling with a simple rule-based agent (no API needed)
This shows how the evaluation framework works without requiring API keys.
"""
import textarena as ta
from test_time_scaling_examples import (
    SelfConsistencyWrapper,
    BestOfNWrapper,
    IterativeRefinementWrapper,
)
import random


class RandomAgent(ta.Agent):
    """
    Simple random agent for TicTacToe that picks valid moves.
    No API needed - perfect for testing the evaluation framework.
    """
    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def __call__(self, observation: str) -> str:
        """Pick a random valid cell from 0-8"""
        # Very simple: just pick a random cell
        cell = self.rng.randint(0, 8)
        return f"[{cell}]"


class SimpleTicTacToeAgent(ta.Agent):
    """
    Slightly smarter agent that tries to win or block.
    Still no API needed.
    """
    def __init__(self, seed=None):
        super().__init__()
        self.rng = random.Random(seed)

    def __call__(self, observation: str) -> str:
        # Extract board state from observation (simplified)
        # For demo purposes, just pick first available cell mentioned
        import re

        # Look for cell numbers in the observation
        cells = re.findall(r'\b([0-8])\b', observation)

        if cells:
            # Pick a random cell from mentioned ones
            cell = self.rng.choice(cells)
        else:
            # Fallback to random
            cell = self.rng.randint(0, 8)

        return f"[{cell}]"


def demo_single_game():
    """Demo a single game between random agents"""
    print("=" * 60)
    print("DEMO: Single Game (Random vs Random)")
    print("=" * 60)

    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=123)

    env = ta.make("TicTacToe-v0")
    env.reset(num_players=2, seed=1)

    agents = {0: agent1, 1: agent2}

    turn = 0
    done = False

    while not done and turn < 20:  # Safety limit
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        print(f"Turn {turn + 1} - Player {player_id}: {action}")
        done, step_info = env.step(action=action)
        turn += 1

    rewards, game_info = env.close()
    print(f"\nGame finished! Rewards: {rewards}")
    print(f"Winner: Player {0 if rewards[0] > rewards[1] else 1 if rewards[1] > rewards[0] else 'Draw'}")


def demo_evaluation_framework():
    """
    Demo the evaluation framework structure.

    Note: This won't test actual test-time scaling since the agents are
    deterministic/random, but it shows how the framework works.
    """
    print("\n" + "=" * 60)
    print("DEMO: Evaluation Framework Structure")
    print("=" * 60)
    print("\nNote: Using random agents (no API required)")
    print("This demonstrates the framework, not actual scaling benefits.\n")

    # Create agents
    baseline = RandomAgent(seed=42)

    # For demo, wrapping random agent won't help, but shows the pattern
    # In real usage, you'd wrap an LLM agent
    wrapped = RandomAgent(seed=43)  # Different seed to simulate different behavior

    agents_to_test = {
        "baseline": baseline,
        "variant": wrapped,
    }

    n_games = 3

    results = {}

    for name, agent in agents_to_test.items():
        print(f"\nTesting: {name}")
        print("-" * 40)

        wins = 0
        for i in range(n_games):
            # Create opponent
            opponent = RandomAgent(seed=999)

            # Alternate starting positions
            if i % 2 == 0:
                agent1, agent2 = agent, opponent
                player_id = 0
            else:
                agent1, agent2 = opponent, agent
                player_id = 1

            # Play game
            env = ta.make("TicTacToe-v0")
            env.reset(num_players=2, seed=i)

            game_agents = {0: agent1, 1: agent2}

            done = False
            turn = 0
            while not done and turn < 20:
                pid, observation = env.get_observation()
                action = game_agents[pid](observation)
                done, step_info = env.step(action=action)
                turn += 1

            rewards, game_info = env.close()

            if rewards[player_id] > rewards[1-player_id]:
                wins += 1
                result = "WIN"
            elif rewards[player_id] == rewards[1-player_id]:
                result = "DRAW"
            else:
                result = "LOSS"

            print(f"  Game {i+1}: {result}")

        win_rate = wins / n_games
        results[name] = {"wins": wins, "games": n_games, "win_rate": win_rate}
        print(f"  Win rate: {win_rate:.1%}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, stats in results.items():
        print(f"{name:20s}: {stats['wins']}/{stats['games']} ({stats['win_rate']:.1%})")

    print("\n" + "=" * 60)
    print("To use with real LLM agents:")
    print("=" * 60)
    print("""
1. Set your API key:
   export OPENROUTER_API_KEY="your-key"

2. Run the full evaluation:
   python evaluate_scaling_approaches.py \\
       --env TicTacToe-v0 \\
       --n-games 10 \\
       --approaches baseline self_consistency_5

3. The framework will:
   - Test each approach against baseline
   - Track wins/losses/draws
   - Measure time and turns
   - Save results to JSON
   - Display summary table
""")


if __name__ == "__main__":
    print("Test-Time Scaling Framework Demo")
    print("(No API key required - uses random agents)\n")

    # Run demos
    demo_single_game()
    demo_evaluation_framework()
