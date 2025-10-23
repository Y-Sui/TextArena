"""
Test-Time Scaling Approaches for TextArena
===========================================

This module demonstrates various test-time scaling techniques that can be used
to improve LLM performance on TextArena games.

"""

import re
from typing import List, Dict, Any, Optional, Callable
from collections import Counter
import textarena as ta


# ============================================================================
# 1. SELF-CONSISTENCY (Multiple Samples + Majority Vote)
# ============================================================================

class SelfConsistencyWrapper(ta.AgentWrapper):
    """
    Generate multiple independent responses and select the most common one.

    Good for: Games with discrete action spaces (TicTacToe, Chess, etc.)
    Scales with: Number of samples (n_samples)
    """

    def __init__(
        self,
        agent: ta.Agent,
        n_samples: int = 5,
        temperature: float = 0.7,
        debugging: bool = False
    ):
        super().__init__(agent)
        self.n_samples = n_samples
        self.temperature = temperature
        self.debugging = debugging

        # Temporarily modify agent temperature if supported
        if hasattr(agent, 'kwargs'):
            self.original_temp = agent.kwargs.get('temperature', 1.0)
            agent.kwargs['temperature'] = temperature

    def __call__(self, observation: str) -> str:
        responses = []

        # Generate multiple samples
        for i in range(self.n_samples):
            response = self.agent(observation)
            responses.append(response)
            if self.debugging:
                print(f"Sample {i+1}: {response}")

        # Extract actions (text in brackets format: [action])
        actions = []
        for response in responses:
            # Try to extract bracketed action
            match = re.search(r'\[([^\]]+)\]', response)
            if match:
                actions.append(match.group(0))  # Keep brackets
            else:
                actions.append(response)

        # Majority vote
        if actions:
            most_common = Counter(actions).most_common(1)[0][0]
            if self.debugging:
                print(f"\nMajority vote result: {most_common}")
                print(f"Vote distribution: {Counter(actions)}")
            return most_common
        else:
            return responses[0]


# ============================================================================
# 2. BEST-OF-N SAMPLING (Generate N, Select Best via Self-Evaluation)
# ============================================================================

class BestOfNWrapper(ta.AgentWrapper):
    """
    Generate N candidate actions, then use the model to evaluate and select the best.

    Good for: Strategic games where quality > diversity
    Scales with: Number of candidates (n_candidates)
    """

    def __init__(
        self,
        agent: ta.Agent,
        n_candidates: int = 3,
        evaluation_prompt: Optional[str] = None,
        debugging: bool = False
    ):
        super().__init__(agent)
        self.n_candidates = n_candidates
        self.debugging = debugging

        self.evaluation_prompt = evaluation_prompt or (
            "You are evaluating different possible actions in a game. "
            "Given the game state and {n} candidate actions, select the BEST action "
            "that maximizes your chance of winning. Respond with ONLY the number (1-{n}) "
            "of the best action."
        )

    def __call__(self, observation: str) -> str:
        # Generate candidate actions
        candidates = []
        for i in range(self.n_candidates):
            action = self.agent(observation)
            candidates.append(action)
            if self.debugging:
                print(f"Candidate {i+1}: {action}")

        # Create evaluation prompt
        eval_prompt = f"{observation}\n\nCandidate actions:\n"
        for i, action in enumerate(candidates, 1):
            eval_prompt += f"{i}. {action}\n"
        eval_prompt += f"\n{self.evaluation_prompt.format(n=self.n_candidates)}"

        # Get evaluation
        original_prompt = self.agent.system_prompt
        self.agent.system_prompt = "You are an expert game-playing evaluator."

        evaluation = self.agent(eval_prompt)
        self.agent.system_prompt = original_prompt

        if self.debugging:
            print(f"\nEvaluation response: {evaluation}")

        # Extract selection (looking for number 1-n)
        for i in range(1, self.n_candidates + 1):
            if str(i) in evaluation[:10]:  # Check first 10 chars
                if self.debugging:
                    print(f"Selected candidate {i}: {candidates[i-1]}")
                return candidates[i-1]

        # Default to first candidate if parsing fails
        if self.debugging:
            print(f"Parsing failed, defaulting to first candidate")
        return candidates[0]


# ============================================================================
# 3. ITERATIVE REFINEMENT (Generate → Critique → Refine)
# ============================================================================

class IterativeRefinementWrapper(ta.AgentWrapper):
    """
    Generate initial action, critique it, then refine based on critique.

    Good for: Complex strategic decisions
    Scales with: Number of refinement iterations
    """

    def __init__(
        self,
        agent: ta.Agent,
        n_iterations: int = 2,
        critique_prompt: Optional[str] = None,
        refinement_prompt: Optional[str] = None,
        debugging: bool = False
    ):
        super().__init__(agent)
        self.n_iterations = n_iterations
        self.debugging = debugging

        self.critique_prompt = critique_prompt or (
            "Critique the following action in the context of the game. "
            "What are its strengths and weaknesses? What could be improved?"
        )

        self.refinement_prompt = refinement_prompt or (
            "Based on the critique, provide an improved action. "
            "Address the weaknesses identified while maintaining the strengths."
        )

    def __call__(self, observation: str) -> str:
        # Initial action
        current_action = self.agent(observation)
        if self.debugging:
            print(f"Initial action: {current_action}")

        original_prompt = self.agent.system_prompt

        # Iterative refinement
        for iteration in range(self.n_iterations):
            # Critique phase
            self.agent.system_prompt = "You are a critical game analyst."
            critique_input = f"{observation}\n\nProposed action: {current_action}\n\n{self.critique_prompt}"
            critique = self.agent(critique_input)

            if self.debugging:
                print(f"\nIteration {iteration + 1} - Critique: {critique}")

            # Refinement phase
            self.agent.system_prompt = original_prompt
            refinement_input = (
                f"{observation}\n\n"
                f"Previous action: {current_action}\n"
                f"Critique: {critique}\n\n"
                f"{self.refinement_prompt}"
            )
            current_action = self.agent(refinement_input)

            if self.debugging:
                print(f"Iteration {iteration + 1} - Refined action: {current_action}")

        self.agent.system_prompt = original_prompt
        return current_action


# ============================================================================
# 4. CHAIN-OF-THOUGHT WITH VERIFICATION
# ============================================================================

class CoTVerificationWrapper(ta.AgentWrapper):
    """
    Generate chain-of-thought reasoning, then verify the final answer makes sense.

    Good for: Games requiring logical reasoning
    Scales with: Verification iterations
    """

    def __init__(
        self,
        agent: ta.Agent,
        max_verification_attempts: int = 2,
        debugging: bool = False
    ):
        super().__init__(agent)
        self.max_verification_attempts = max_verification_attempts
        self.debugging = debugging

    def __call__(self, observation: str) -> str:
        original_prompt = self.agent.system_prompt

        # Generate CoT reasoning
        cot_prompt = (
            "Think step-by-step about the best action. "
            "Analyze the current state, consider your options, "
            "evaluate each option, then provide your final answer after 'Final Answer:'"
        )
        self.agent.system_prompt = original_prompt + "\n\n" + cot_prompt

        reasoning = self.agent(observation)
        if self.debugging:
            print(f"Reasoning:\n{reasoning}\n")

        # Extract answer
        if "Final Answer:" in reasoning:
            action = reasoning.split("Final Answer:")[-1].strip()
        else:
            action = reasoning

        # Verification phase
        for attempt in range(self.max_verification_attempts):
            verification_prompt = (
                f"Given this game state:\n{observation}\n\n"
                f"And this reasoning:\n{reasoning}\n\n"
                f"Is the action '{action}' valid and optimal? "
                f"Respond with 'VALID' if yes, or suggest a better action if no."
            )

            self.agent.system_prompt = "You are a game move validator."
            verification = self.agent(verification_prompt)

            if self.debugging:
                print(f"Verification {attempt + 1}: {verification}")

            if "VALID" in verification.upper():
                break
            else:
                # Extract suggested action from verification
                match = re.search(r'\[([^\]]+)\]', verification)
                if match:
                    action = match.group(0)
                    if self.debugging:
                        print(f"Updated action: {action}")

        self.agent.system_prompt = original_prompt
        return action


# ============================================================================
# 5. ENSEMBLE WITH DIFFERENT STRATEGIES
# ============================================================================

class EnsembleStrategyWrapper(ta.AgentWrapper):
    """
    Use multiple prompting strategies and combine their outputs.

    Good for: Complex games where different approaches have merit
    Scales with: Number of strategies
    """

    def __init__(
        self,
        agent: ta.Agent,
        strategies: Optional[List[str]] = None,
        combination_method: str = "vote",  # "vote" or "evaluate"
        debugging: bool = False
    ):
        super().__init__(agent)
        self.combination_method = combination_method
        self.debugging = debugging

        self.strategies = strategies or [
            "Play aggressively to maximize immediate advantage.",
            "Play defensively to minimize opponent's options.",
            "Play strategically for long-term position.",
        ]

    def __call__(self, observation: str) -> str:
        original_prompt = self.agent.system_prompt
        actions = []

        # Generate action with each strategy
        for i, strategy in enumerate(self.strategies):
            self.agent.system_prompt = original_prompt + f"\n\nStrategy: {strategy}"
            action = self.agent(observation)
            actions.append(action)

            if self.debugging:
                print(f"Strategy {i+1} ({strategy}): {action}")

        # Combine actions
        if self.combination_method == "vote":
            # Majority vote
            final_action = Counter(actions).most_common(1)[0][0]
        else:  # evaluate
            # Use model to select best
            eval_prompt = f"{observation}\n\nProposed actions from different strategies:\n"
            for i, (strategy, action) in enumerate(zip(self.strategies, actions), 1):
                eval_prompt += f"{i}. [{strategy}] {action}\n"
            eval_prompt += "\nWhich action is best? Respond with the number (1-{}).".format(len(actions))

            self.agent.system_prompt = "You are an expert game evaluator."
            evaluation = self.agent(eval_prompt)

            # Parse selection
            for i in range(1, len(actions) + 1):
                if str(i) in evaluation[:10]:
                    final_action = actions[i-1]
                    break
            else:
                final_action = actions[0]

        self.agent.system_prompt = original_prompt

        if self.debugging:
            print(f"\nFinal action: {final_action}")

        return final_action


# ============================================================================
# 6. MONTE CARLO TREE SEARCH (MCTS) - Simplified
# ============================================================================

class SimplifiedMCTSWrapper(ta.AgentWrapper):
    """
    Simplified MCTS: Generate multiple action sequences, evaluate them, pick best first move.

    Good for: Turn-based games with clear win conditions
    Scales with: Number of rollouts × rollout depth

    Note: This is a simplified version. Full MCTS requires environment simulation.
    """

    def __init__(
        self,
        agent: ta.Agent,
        n_rollouts: int = 3,
        rollout_depth: int = 3,
        debugging: bool = False
    ):
        super().__init__(agent)
        self.n_rollouts = n_rollouts
        self.rollout_depth = rollout_depth
        self.debugging = debugging

    def __call__(self, observation: str) -> str:
        original_prompt = self.agent.system_prompt
        candidate_actions = {}

        # Simulate multiple rollouts
        for rollout in range(self.n_rollouts):
            # Generate initial action
            first_action = self.agent(observation)

            # Simulate future states (imagined)
            self.agent.system_prompt = (
                "Imagine the game state after your action. "
                "Describe likely outcomes and whether this path leads to victory. "
                "Rate the position from 0-10 where 10 is certain victory."
            )

            simulation_prompt = (
                f"{observation}\n\n"
                f"If I take action: {first_action}\n"
                f"Simulate {self.rollout_depth} future moves and evaluate the final position."
            )

            simulation = self.agent(simulation_prompt)

            # Extract score
            score_match = re.search(r'(\d+)/10|(\d+)\s*out of\s*10|score:\s*(\d+)', simulation.lower())
            if score_match:
                score = int([g for g in score_match.groups() if g][0])
            else:
                score = 5  # Default neutral score

            if first_action in candidate_actions:
                candidate_actions[first_action].append(score)
            else:
                candidate_actions[first_action] = [score]

            if self.debugging:
                print(f"Rollout {rollout + 1}: {first_action} -> Score: {score}")

        # Select action with best average score
        best_action = max(
            candidate_actions.items(),
            key=lambda x: sum(x[1]) / len(x[1])
        )[0]

        self.agent.system_prompt = original_prompt

        if self.debugging:
            print(f"\nAction scores: {candidate_actions}")
            print(f"Selected: {best_action}")

        return best_action


# ============================================================================
# 7. TEMPERATURE LADDER (Sample at different temperatures)
# ============================================================================

class TemperatureLadderWrapper(ta.AgentWrapper):
    """
    Generate actions at different temperature settings and ensemble them.

    Good for: Balancing exploration and exploitation
    Scales with: Number of temperature levels
    """

    def __init__(
        self,
        agent: ta.Agent,
        temperatures: Optional[List[float]] = None,
        selection_method: str = "vote",  # "vote", "median", or "highest_temp"
        debugging: bool = False
    ):
        super().__init__(agent)
        self.temperatures = temperatures or [0.3, 0.7, 1.0]
        self.selection_method = selection_method
        self.debugging = debugging

        # Save original temperature
        if hasattr(agent, 'kwargs'):
            self.original_temp = agent.kwargs.get('temperature', 1.0)

    def __call__(self, observation: str) -> str:
        actions = []

        for temp in self.temperatures:
            # Set temperature
            if hasattr(self.agent, 'kwargs'):
                self.agent.kwargs['temperature'] = temp

            action = self.agent(observation)
            actions.append(action)

            if self.debugging:
                print(f"Temperature {temp}: {action}")

        # Restore original temperature
        if hasattr(self.agent, 'kwargs'):
            self.agent.kwargs['temperature'] = self.original_temp

        # Select final action
        if self.selection_method == "vote":
            final_action = Counter(actions).most_common(1)[0][0]
        elif self.selection_method == "median":
            final_action = actions[len(actions) // 2]
        else:  # highest_temp
            final_action = actions[-1]

        if self.debugging:
            print(f"Selected action: {final_action}")

        return final_action


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """
    Example of how to use these test-time scaling wrappers
    """
    import textarena as ta

    # Create base agent
    base_agent = ta.agents.OpenRouterAgent(model_name="anthropic/claude-3.5-haiku")

    # Wrap with test-time scaling approach
    # Option 1: Self-consistency
    agent_sc = SelfConsistencyWrapper(base_agent, n_samples=3, debugging=True)

    # Option 2: Best-of-N
    agent_bon = BestOfNWrapper(base_agent, n_candidates=3, debugging=True)

    # Option 3: Iterative refinement
    agent_refine = IterativeRefinementWrapper(base_agent, n_iterations=2, debugging=True)

    # Option 4: Chain multiple wrappers
    agent_complex = SelfConsistencyWrapper(
        IterativeRefinementWrapper(base_agent, n_iterations=1),
        n_samples=3
    )

    # Use in game
    agents = {
        0: agent_sc,  # Use scaling agent
        1: base_agent  # Baseline agent
    }

    env = ta.make("TicTacToe-v0")
    env = ta.wrappers.SimpleRenderWrapper(env=env)
    env.reset(num_players=2)

    done = False
    while not done:
        player_id, observation = env.get_observation()
        action = agents[player_id](observation)
        done, step_info = env.step(action=action)

    rewards, game_info = env.close()
    print(f"\nGame finished! Rewards: {rewards}")


if __name__ == "__main__":
    # print("Test-Time Scaling Wrappers for TextArena")
    # print("=" * 50)
    # print("\nAvailable wrappers:")
    # print("1. SelfConsistencyWrapper - Multiple samples + majority vote")
    # print("2. BestOfNWrapper - Generate N candidates, select best via evaluation")
    # print("3. IterativeRefinementWrapper - Generate → Critique → Refine")
    # print("4. CoTVerificationWrapper - Chain-of-thought with verification")
    # print("5. EnsembleStrategyWrapper - Multiple strategies combined")
    # print("6. SimplifiedMCTSWrapper - Simplified Monte Carlo Tree Search")
    # print("7. TemperatureLadderWrapper - Sample at different temperatures")
    # print("\nSee example_usage() for how to use these wrappers.")
    
    example_usage()
