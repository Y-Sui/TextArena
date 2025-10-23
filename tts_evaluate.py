"""
Evaluation Script for Test-Time Scaling Approaches
===================================================

This script helps you systematically evaluate different test-time scaling
approaches on TextArena benchmarks.

Usage:
    python evaluate_scaling_approaches.py --env TicTacToe-v0 --agent openrouter --n-games 3
"""

import argparse
import json
import time
import weave
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import textarena as ta
from tts_baselines import (
    SelfConsistencyWrapper,
    BestOfNWrapper,
    IterativeRefinementWrapper,
    CoTVerificationWrapper,
    EnsembleStrategyWrapper,
    SimplifiedMCTSWrapper,
    TemperatureLadderWrapper,
)
from test_api_connection import test_api_connection

class ScalingEvaluator:
    """
    Evaluates test-time scaling approaches on TextArena games.
    """

    def __init__(
        self,
        base_agent_type: str = "openrouter",
        model_name: str = "anthropic/claude-3.5-haiku",
        results_dir: str = "scaling_results"
    ):
        self.base_agent_type = base_agent_type
        self.model_name = model_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

    def create_base_agent(self) -> ta.Agent:
        """Create a base agent for testing."""
        if self.base_agent_type == "openrouter":
            return ta.agents.OpenRouterAgent(model_name=self.model_name)
        elif self.base_agent_type == "openai":
            return ta.agents.OpenAIAgent(model_name=self.model_name)
        elif self.base_agent_type == "anthropic":
            return ta.agents.AnthropicAgent(model_name=self.model_name)
        else:
            raise ValueError(f"Unknown agent type: {self.base_agent_type}")

    def create_scaling_agents(self, base_agent: ta.Agent) -> Dict[str, ta.Agent]:
        """
        Create different scaling approach agents.

        Returns:
            Dict mapping approach name to wrapped agent
        """
        return {
            # "baseline": base_agent,
            "self_consistency_3": SelfConsistencyWrapper(
                self.create_base_agent(), n_samples=3, debugging=False
            ),
            "self_consistency_5": SelfConsistencyWrapper(
                self.create_base_agent(), n_samples=5, debugging=False
            ),
            "best_of_3": BestOfNWrapper(
                self.create_base_agent(), n_candidates=3, debugging=False
            ),
            "iterative_refinement_1": IterativeRefinementWrapper(
                self.create_base_agent(), n_iterations=1, debugging=False
            ),
            "iterative_refinement_2": IterativeRefinementWrapper(
                self.create_base_agent(), n_iterations=2, debugging=False
            ),
            "cot_verification": CoTVerificationWrapper(
                self.create_base_agent(), max_verification_attempts=1, debugging=False
            ),
            "ensemble_strategy": EnsembleStrategyWrapper(
                self.create_base_agent(), debugging=False
            ),
            "mcts_3x3": SimplifiedMCTSWrapper(
                self.create_base_agent(), n_rollouts=3, rollout_depth=3, debugging=False
            ),
            "temperature_ladder": TemperatureLadderWrapper(
                self.create_base_agent(), temperatures=[0.3, 0.7, 1.0], debugging=False
            ),
        }

    def run_single_game(
        self,
        env_id: str,
        agent1: ta.Agent,
        agent2: ta.Agent,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run a single game between two agents.

        Returns:
            Dict with game results including rewards, winner, turns, and time
        """
        env = ta.make(env_id)
        env.reset(num_players=2, seed=seed)

        agents = {0: agent1, 1: agent2}

        start_time = time.time()
        turn_count = 0
        done = False

        while not done:
            player_id, observation = env.get_observation()
            action = agents[player_id](observation)
            done, step_info = env.step(action=action)
            turn_count += 1

            # Safety limit to prevent infinite games
            if turn_count > 100:
                print(f"Warning: Game exceeded 1000 turns, stopping early")
                break

        rewards, game_info = env.close()
        elapsed_time = time.time() - start_time

        # Determine winner
        if rewards[0] > rewards[1]:
            winner = 0
        elif rewards[1] > rewards[0]:
            winner = 1
        else:
            winner = None  # Draw

        return {
            "rewards": rewards,
            "winner": winner,
            "turns": turn_count,
            "time_seconds": elapsed_time,
            "game_info": game_info
        }

    def evaluate_approach(
        self,
        env_id: str,
        approach_name: str,
        agent: ta.Agent,
        opponent: Optional[ta.Agent] = None,
        n_games: int = 10,
        seeds: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a scaling approach against a baseline opponent.

        Args:
            env_id: Environment ID to test on
            approach_name: Name of the scaling approach
            agent: The agent using the scaling approach
            n_games: Number of games to play
            opponent: Opponent agent (uses baseline if None)
            seeds: List of seeds for reproducibility

        Returns:
            Dict with evaluation results
        """
        if opponent is None:
            opponent = self.create_base_agent()

        if seeds is None:
            seeds = list(range(n_games))

        print(f"\n{'='*60}")
        print(f"Evaluating: {approach_name} on {env_id}")
        print(f"Playing {n_games} games...")
        print(f"{'='*60}")

        results = []
        wins = 0
        draws = 0
        total_time = 0
        total_turns = 0

        for i, seed in enumerate(seeds[:n_games]):
            print(f"Game {i+1}/{n_games}...", end=" ")

            # Alternate starting positions
            if i % 2 == 0:
                agent1, agent2 = agent, opponent
                player_id = 0
            else:
                agent1, agent2 = opponent, agent
                player_id = 1

            game_result = self.run_single_game(env_id, agent1, agent2, seed=seed)
            results.append(game_result)

            # Track statistics
            if game_result["winner"] == player_id:
                wins += 1
                print("WIN")
            elif game_result["winner"] is None:
                draws += 1
                print("DRAW")
            else:
                print("LOSS")

            total_time += game_result["time_seconds"]
            total_turns += game_result["turns"]

        # Calculate statistics
        win_rate = wins / n_games
        draw_rate = draws / n_games
        avg_time = total_time / n_games
        avg_turns = total_turns / n_games

        summary = {
            "approach": approach_name,
            "env_id": env_id,
            "n_games": n_games,
            "wins": wins,
            "draws": draws,
            "losses": n_games - wins - draws,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "avg_time_seconds": avg_time,
            "avg_turns": avg_turns,
            "total_time_seconds": total_time,
            "results": results
        }

        print(f"\nResults: {wins}W-{draws}D-{n_games-wins-draws}L (Win rate: {win_rate:.1%})")
        print(f"Avg time: {avg_time:.1f}s, Avg turns: {avg_turns:.1f}")

        return summary

    def run_full_evaluation(
        self,
        env_ids: List[str],
        mode: int = 1,
        n_games_per_env: int = 10,
        approaches: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation across multiple environments and approaches.

        Args:
            env_ids: List of environment IDs to test
            n_games_per_env: Number of games per environment
            approaches: List of approach names to test (None = all)

        Returns:
            Complete evaluation results
        """
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_agent_type": self.base_agent_type,
                "model_name": self.model_name,
                "n_games_per_env": n_games_per_env,
            },
            "evaluations": []
        }

        # Create all scaling agents
        scaling_agents = self.create_scaling_agents(self.create_base_agent())

        # Filter approaches if specified
        if approaches:
            scaling_agents = {k: v for k, v in scaling_agents.items() if k in approaches}
            
        # mode 1: compare different scaling approaches against baseline (default agent without TTS prompting)
        if mode == 1:
            opponent = self.create_base_agent()
            for env_id in env_ids:
                for approach_name, agent in scaling_agents.items():
                    try:
                        result = self.evaluate_approach(
                            env_id=env_id,
                            approach_name=approach_name,
                            agent=agent,
                            opponent=opponent,
                            n_games=n_games_per_env
                        )
                        all_results["evaluations"].append(result)

                    except Exception as e:
                        print(f"Error evaluating {approach_name} on {env_id}: {e}")
                        all_results["evaluations"].append({
                            "approach": approach_name,
                            "env_id": env_id,
                            "error": str(e)
                        })
                        
        # mode 2: compare propsed scaling approache against all other TTS opponent (e.g., best_of_5)
        elif mode == 2:
            # define your proposed scaling approach here
            agent = BestOfNWrapper(
                self.create_base_agent(), n_candidates=5, debugging=False
            )
            for env_id in env_ids:
                for approach_name, opponent in scaling_agents.items():
                    try:
                        result = self.evaluate_approach(
                            env_id=env_id,
                            approach_name=approach_name,
                            agent=agent,
                            opponent=opponent,
                            n_games=n_games_per_env
                        )
                        all_results["evaluations"].append(result)

                    except Exception as e:
                        print(f"Error evaluating {approach_name} on {env_id}: {e}")
                        all_results["evaluations"].append({
                            "approach": approach_name,
                            "env_id": env_id,
                            "error": str(e)
                        })
            

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"evaluation_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Evaluation complete! Results saved to: {results_file}")
        print(f"{'='*60}")

        # Print summary table
        self.print_summary_table(all_results)

        return all_results

    def print_summary_table(self, results: Dict[str, Any]):
        """Print a summary table of results."""
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)

        # Group by environment
        envs = {}
        for eval_result in results["evaluations"]:
            if "error" in eval_result:
                continue

            env_id = eval_result["env_id"]
            if env_id not in envs:
                envs[env_id] = []
            envs[env_id].append(eval_result)

        # Print table for each environment
        for env_id, eval_results in envs.items():
            print(f"\n{env_id}")
            print("-" * 80)
            print(f"{'Approach':<30} {'Win Rate':<12} {'Avg Time':<12} {'Avg Turns':<12}")
            print("-" * 80)

            # Sort by win rate
            eval_results.sort(key=lambda x: x["win_rate"], reverse=True)

            for result in eval_results:
                print(
                    f"{result['approach']:<30} "
                    f"{result['win_rate']:>6.1%}      "
                    f"{result['avg_time_seconds']:>6.1f}s      "
                    f"{result['avg_turns']:>6.1f}"
                )

        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate test-time scaling approaches on TextArena"
    )
    parser.add_argument(
        "--env",
        type=str,
        nargs="+",
        default=["TicTacToe-v0"],
        help="Environment ID(s) to test on"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="openrouter",
        choices=["openrouter", "openai", "anthropic"],
        help="Base agent type"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3.5-haiku",
        help="Model name"
    )
    parser.add_argument(
        "--n-games",
        type=int,
        default=3,
        help="Number of games per environment"
    )
    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=None,
        help="Specific approaches to test (default: all)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="scaling_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=[1, 2],
        help="Evaluation mode: 1 = compare scaling approaches against baseline; 2 = compare proposed approach against all others"
    )

    args = parser.parse_args()
    
    weave.init("game_agent")
    
    # test api connnection
    if test_api_connection():
        
        # Create evaluator
        evaluator = ScalingEvaluator(
            base_agent_type=args.agent,
            model_name=args.model,
            results_dir=args.results_dir
        )

        # Run evaluation
        evaluator.run_full_evaluation(
            env_ids=args.env,
            n_games_per_env=args.n_games,
            approaches=args.approaches,
            mode=1  # Change to 2 to evaluate proposed approach against all others
        )
    else:
        print("API connection failed. Please check your API key and try again.")


if __name__ == "__main__":
    main()
