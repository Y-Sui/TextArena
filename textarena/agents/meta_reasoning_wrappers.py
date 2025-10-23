"""
Meta-Reasoning Agent Wrappers for TextArena
============================================

This module implements advanced agent architectures that incorporate:
1. Meta-reasoning with coach and player agents
2. World model simulation for action forecasting

These wrappers enable more sophisticated decision-making through hierarchical
reasoning and environment simulation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import textarena as ta
import weave


# ============================================================================
# 1. META-REASONING AGENT (Coach + Player Architecture)
# ============================================================================

class MetaReasoningAgentWrapper(ta.AgentWrapper):
    """
    Two-level agent architecture with hierarchical reasoning:

    - High-level Coach Agent: Provides strategic guidance, overall tactics,
      and game-level strategy without worrying about specific action formats
    - Low-level Player Agent: Learns game rules, action formats, and executes
      specific moves based on coach's guidance

    This separation allows:
    - Coach to focus on "what to achieve" (strategy, tactics)
    - Player to focus on "how to achieve it" (execution, rules, formats)

    Good for: Complex strategic games requiring both high-level planning
              and low-level execution (Chess, Settlers of Catan, etc.)
    Scales with: Quality and separation of coach/player prompts

    Args:
        agent: The low-level player agent (executes actions)
        coach_agent: The high-level coach agent (provides strategy)
        coach_prompt: Custom prompt for the coach role
        player_incorporation_prompt: How player should use coach's guidance
        debugging: Enable detailed logging
        weave_tracking: Enable weave operation tracking
    """

    def __init__(
        self,
        agent: ta.Agent,
        coach_agent: ta.Agent,
        coach_prompt: Optional[str] = None,
        player_incorporation_prompt: Optional[str] = None,
        debugging: bool = False,
        weave_tracking: bool = True
    ):
        super().__init__(agent)
        self.coach_agent = coach_agent
        self.player_agent = agent
        self.debugging = debugging
        self.weave_tracking = weave_tracking

        # Store original prompts
        self.player_original_prompt = self.player_agent.system_prompt
        self.coach_original_prompt = self.coach_agent.system_prompt

        # Coach focuses on high-level strategy
        self.coach_prompt = coach_prompt or (
            "You are a strategic game coach providing high-level guidance. "
            "Your role is to:\n"
            "1. Analyze the overall game state and strategic position\n"
            "2. Identify key tactical opportunities and threats\n"
            "3. Suggest general strategic directions (e.g., 'control the center', "
            "'play defensively', 'target opponent's weak resources')\n"
            "4. Focus on the 'what' and 'why', not specific move syntax\n\n"
            "Provide clear, actionable strategic guidance in 2-4 sentences. "
            "Do NOT specify exact moves or action formats - that's the player's job."
        )

        # Player incorporates coach's guidance
        self.player_incorporation_prompt = player_incorporation_prompt or (
            "\n\n=== COACH'S STRATEGIC GUIDANCE ===\n{guidance}\n"
            "=== YOUR TASK ===\n"
            "Based on the coach's strategic guidance above and the game state, "
            "execute a specific action that follows this strategy. "
            "Make sure to follow all game rules and use the correct action format."
        )

        # Track history for learning
        self.interaction_history = []

    def _get_coach_guidance(self, observation: str) -> str:
        """Get strategic guidance from the coach agent."""
        # Set coach-specific prompt
        self.coach_agent.system_prompt = self.coach_prompt

        # Add recent history context if available
        context = ""
        if self.interaction_history:
            recent = self.interaction_history[-3:]  # Last 3 interactions
            context = "\n\nRECENT GAME HISTORY:\n"
            for i, (strat, action, outcome) in enumerate(recent, 1):
                context += f"{i}. Strategy: {strat[:100]}... -> Action taken\n"

        # Get coach's strategic analysis
        coach_input = f"{observation}{context}\n\nProvide strategic guidance:"
        guidance = self.coach_agent(coach_input)

        # Restore original prompt
        self.coach_agent.system_prompt = self.coach_original_prompt

        return guidance

    def _execute_with_guidance(self, observation: str, guidance: str) -> str:
        """Execute action based on coach's guidance."""
        # Prepare player prompt with integrated guidance
        player_prompt = self.player_original_prompt + self.player_incorporation_prompt.format(
            guidance=guidance
        )
        self.player_agent.system_prompt = player_prompt

        # Player executes based on guidance
        action = self.player_agent(observation)

        # Restore original prompt
        self.player_agent.system_prompt = self.player_original_prompt

        return action

    @weave.op()
    def __call__(self, observation: str) -> str:
        """
        Process observation through meta-reasoning:
        1. Coach analyzes and provides strategic guidance
        2. Player executes specific action based on guidance
        """
        if self.debugging:
            print("\n" + "="*70)
            print("META-REASONING AGENT")
            print("="*70)
            print("\n[OBSERVATION]")
            print(observation[:200] + "..." if len(observation) > 200 else observation)

        # Step 1: Coach provides high-level strategic guidance
        guidance = self._get_coach_guidance(observation)

        if self.debugging:
            print("\n[COACH'S STRATEGIC GUIDANCE]")
            print(guidance)

        # Step 2: Player executes based on guidance
        action = self._execute_with_guidance(observation, guidance)

        if self.debugging:
            print("\n[PLAYER'S ACTION]")
            print(action)
            print("="*70 + "\n")

        # Store interaction for future learning
        self.interaction_history.append((guidance, action, observation))

        # Keep history manageable
        if len(self.interaction_history) > 10:
            self.interaction_history = self.interaction_history[-10:]

        return action

    def reset_history(self):
        """Reset interaction history (call between games)."""
        self.interaction_history = []


# ============================================================================
# 2. WORLD MODEL AGENT (Environment Simulation & Forecasting)
# ============================================================================

class WorldModelAgentWrapper(ta.AgentWrapper):
    """
    Agent with internal world model for action impact forecasting:

    - Generates candidate actions
    - Simulates each action's impact on the environment
    - Forecasts outcomes using learned/simulated experience
    - Selects action with best predicted outcome

    This enables "what-if" reasoning: "If I do X, what happens to the game state?"

    The world model learns patterns through:
    1. Prior experience (action history and outcomes)
    2. Simulated rollouts (imagined future states)
    3. Multi-step forecasting (predict opponent responses)

    Good for: Games with complex state transitions and indirect effects
              (e.g., Catan resource trading, Chess tactical sequences)
    Scales with: Number of candidate actions × simulation depth

    Args:
        agent: The base agent for action generation
        simulator_agent: Agent used for world model simulation (can be same as agent)
        n_candidates: Number of candidate actions to consider
        simulation_depth: How many steps ahead to simulate
        use_action_history: Whether to use past actions as prior experience
        simulation_prompt: Custom prompt for environment simulation
        evaluation_prompt: Custom prompt for outcome evaluation
        debugging: Enable detailed logging
        weave_tracking: Enable weave operation tracking
    """

    def __init__(
        self,
        agent: ta.Agent,
        simulator_agent: Optional[ta.Agent] = None,
        n_candidates: int = 3,
        simulation_depth: int = 2,
        use_action_history: bool = True,
        simulation_prompt: Optional[str] = None,
        evaluation_prompt: Optional[str] = None,
        debugging: bool = False,
        weave_tracking: bool = True
    ):
        super().__init__(agent)
        self.simulator_agent = simulator_agent or agent
        self.n_candidates = n_candidates
        self.simulation_depth = simulation_depth
        self.use_action_history = use_action_history
        self.debugging = debugging
        self.weave_tracking = weave_tracking

        # Store original prompts
        self.agent_original_prompt = self.agent.system_prompt
        self.simulator_original_prompt = self.simulator_agent.system_prompt

        # World model simulation prompt
        self.simulation_prompt = simulation_prompt or (
            "You are a world model that simulates game environments.\n\n"
            "Given a current game state and a proposed action, forecast what will happen:\n"
            "1. How does the game state change immediately?\n"
            "2. How might opponents respond?\n"
            "3. What is the likely state after {depth} moves?\n"
            "4. Rate the outcome favorability from 0-10 (10 = very favorable)\n\n"
            "Focus on concrete state changes and strategic implications."
        )

        # Outcome evaluation prompt
        self.evaluation_prompt = evaluation_prompt or (
            "You are evaluating simulated game outcomes.\n\n"
            "Compare the forecasted outcomes and select the action that:\n"
            "1. Maximizes winning probability\n"
            "2. Improves strategic position\n"
            "3. Minimizes risks\n\n"
            "Respond with ONLY the number (1-{n}) of the best action."
        )

        # Prior experience storage
        self.action_history: List[Dict[str, Any]] = []
        self.simulation_cache: Dict[str, Dict[str, Any]] = {}

    def _generate_candidate_actions(self, observation: str) -> List[str]:
        """Generate multiple candidate actions to evaluate."""
        candidates = []

        # Adjust temperature for diversity if possible
        original_temp = None
        if hasattr(self.agent, 'kwargs') and 'temperature' in self.agent.kwargs:
            original_temp = self.agent.kwargs['temperature']
            self.agent.kwargs['temperature'] = 0.8  # Higher temp for diversity

        # Generate diverse candidates
        for i in range(self.n_candidates):
            action = self.agent(observation)
            candidates.append(action)

        # Restore temperature
        if original_temp is not None:
            self.agent.kwargs['temperature'] = original_temp

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for action in candidates:
            # Normalize for comparison (extract core action)
            normalized = re.sub(r'\s+', ' ', action.strip().lower())
            if normalized not in seen:
                seen.add(normalized)
                unique_candidates.append(action)

        return unique_candidates

    def _build_prior_experience_context(self) -> str:
        """Build context from prior action history."""
        if not self.use_action_history or not self.action_history:
            return ""

        context = "\n\nPRIOR EXPERIENCE (recent actions and outcomes):\n"
        recent = self.action_history[-5:]  # Last 5 actions

        for i, record in enumerate(recent, 1):
            action = record['action']
            outcome_score = record.get('outcome_score', 'N/A')
            outcome_summary = record.get('outcome_summary', '')[:100]
            context += f"{i}. Action: {action[:50]}... Score: {outcome_score}\n"
            if outcome_summary:
                context += f"   Outcome: {outcome_summary}...\n"

        return context

    def _simulate_action_outcome(
        self,
        observation: str,
        action: str,
        prior_context: str
    ) -> Dict[str, Any]:
        """
        Simulate what happens if this action is taken.
        Returns predicted outcome and favorability score.
        """
        # Check cache first
        cache_key = f"{hash(observation)}_{hash(action)}"
        if cache_key in self.simulation_cache:
            if self.debugging:
                print(f"  [Using cached simulation for: {action[:50]}...]")
            return self.simulation_cache[cache_key]

        # Set simulator prompt
        self.simulator_agent.system_prompt = self.simulation_prompt.format(
            depth=self.simulation_depth
        )

        # Build simulation query
        simulation_query = (
            f"CURRENT GAME STATE:\n{observation}\n"
            f"{prior_context}\n"
            f"PROPOSED ACTION: {action}\n\n"
            f"Simulate the next {self.simulation_depth} moves. "
            f"What happens? Rate the outcome (0-10):"
        )

        # Run simulation
        simulation_result = self.simulator_agent(simulation_query)

        # Restore original prompt
        self.simulator_agent.system_prompt = self.simulator_original_prompt

        # Extract favorability score
        score_match = re.search(
            r'(?:score|rating|favorability)[:=\s]*(\d+(?:\.\d+)?)\s*[/]?\s*10',
            simulation_result.lower()
        )
        if score_match:
            score = float(score_match.group(1))
        else:
            # Try to find any number 0-10
            numbers = re.findall(r'\b([0-9]|10)(?:\.\d+)?\b', simulation_result)
            score = float(numbers[-1]) if numbers else 5.0  # Default to neutral

        result = {
            'action': action,
            'simulation': simulation_result,
            'score': score
        }

        # Cache result
        self.simulation_cache[cache_key] = result

        # Keep cache size manageable
        if len(self.simulation_cache) > 100:
            # Remove oldest entries
            remove_keys = list(self.simulation_cache.keys())[:50]
            for key in remove_keys:
                del self.simulation_cache[key]

        return result

    def _select_best_action(
        self,
        simulated_outcomes: List[Dict[str, Any]]
    ) -> str:
        """Select the best action based on simulated outcomes."""
        # Sort by score
        sorted_outcomes = sorted(
            simulated_outcomes,
            key=lambda x: x['score'],
            reverse=True
        )

        if self.debugging:
            print("\n[OUTCOME RANKINGS]")
            for i, outcome in enumerate(sorted_outcomes, 1):
                action_preview = outcome['action'][:60]
                print(f"{i}. Score {outcome['score']:.1f}/10: {action_preview}...")

        # Return highest-scored action
        return sorted_outcomes[0]['action']

    @weave.op()
    def __call__(self, observation: str) -> str:
        """
        Process observation through world model:
        1. Generate candidate actions
        2. Simulate each action's impact on the environment
        3. Evaluate outcomes using prior experience
        4. Select action with best predicted outcome
        """
        if self.debugging:
            print("\n" + "="*70)
            print("WORLD MODEL AGENT")
            print("="*70)
            print("\n[OBSERVATION]")
            print(observation[:200] + "..." if len(observation) > 200 else observation)

        # Step 1: Generate candidate actions
        candidates = self._generate_candidate_actions(observation)

        if self.debugging:
            print(f"\n[GENERATED {len(candidates)} CANDIDATE ACTIONS]")
            for i, action in enumerate(candidates, 1):
                print(f"{i}. {action[:70]}...")

        # Step 2: Build prior experience context
        prior_context = self._build_prior_experience_context()

        # Step 3: Simulate each candidate action
        if self.debugging:
            print("\n[SIMULATING ACTION OUTCOMES]")

        simulated_outcomes = []
        for i, action in enumerate(candidates, 1):
            if self.debugging:
                print(f"\nSimulating action {i}/{len(candidates)}...")

            outcome = self._simulate_action_outcome(observation, action, prior_context)
            simulated_outcomes.append(outcome)

            if self.debugging:
                print(f"  Score: {outcome['score']:.1f}/10")
                print(f"  Forecast: {outcome['simulation'][:150]}...")

        # Step 4: Select best action
        best_action = self._select_best_action(simulated_outcomes)

        if self.debugging:
            print(f"\n[SELECTED BEST ACTION]")
            print(best_action)
            print("="*70 + "\n")

        # Step 5: Record for future learning
        self.action_history.append({
            'observation': observation[:500],  # Truncate for memory
            'action': best_action,
            'candidates': candidates,
            'simulations': simulated_outcomes,
            'outcome_score': simulated_outcomes[0]['score']
        })

        # Keep history manageable
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]

        return best_action

    def reset_history(self):
        """Reset action history and simulation cache (call between games)."""
        self.action_history = []
        self.simulation_cache = {}

    def get_learned_patterns(self) -> Dict[str, Any]:
        """
        Extract learned patterns from action history.
        Useful for analysis and debugging.
        """
        if not self.action_history:
            return {"message": "No history available"}

        # Analyze successful actions (score >= 7)
        successful_actions = [
            record for record in self.action_history
            if record.get('outcome_score', 0) >= 7.0
        ]

        # Analyze unsuccessful actions (score < 4)
        unsuccessful_actions = [
            record for record in self.action_history
            if record.get('outcome_score', 10) < 4.0
        ]

        return {
            'total_decisions': len(self.action_history),
            'successful_actions': len(successful_actions),
            'unsuccessful_actions': len(unsuccessful_actions),
            'average_score': sum(
                r.get('outcome_score', 5.0) for r in self.action_history
            ) / len(self.action_history),
            'cache_size': len(self.simulation_cache)
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_meta_reasoning():
    """Example: Meta-reasoning agent with coach and player"""
    import textarena as ta

    # Create base agents
    player_agent = ta.agents.OpenAIAgent(
        model_name="gpt-4o-mini",
        system_prompt="You are a game player. Execute actions precisely."
    )

    coach_agent = ta.agents.OpenAIAgent(
        model_name="gpt-4o",
        system_prompt="You are a strategic coach."
    )

    # Wrap with meta-reasoning
    meta_agent = MetaReasoningAgentWrapper(
        agent=player_agent,
        coach_agent=coach_agent,
        debugging=True
    )

    # Use in game
    env = ta.make("TicTacToe-v0")
    env.reset(num_players=1)

    player_id, observation = env.get_observation()
    action = meta_agent(observation)
    print(f"Action: {action}")


def example_world_model():
    """Example: World model agent with simulation"""
    import textarena as ta

    # Create base agent
    base_agent = ta.agents.OpenAIAgent(
        model_name="gpt-4o",
        system_prompt="You are an expert game player."
    )

    # Create simulator (can be same or different model)
    simulator = ta.agents.OpenAIAgent(
        model_name="gpt-4o-mini",
        system_prompt="You simulate game environments."
    )

    # Wrap with world model
    world_model_agent = WorldModelAgentWrapper(
        agent=base_agent,
        simulator_agent=simulator,
        n_candidates=3,
        simulation_depth=2,
        debugging=True
    )

    # Use in game
    env = ta.make("TicTacToe-v0")
    env.reset(num_players=1)

    player_id, observation = env.get_observation()
    action = world_model_agent(observation)
    print(f"Action: {action}")

    # Check learned patterns after game
    patterns = world_model_agent.get_learned_patterns()
    print(f"Learned patterns: {patterns}")


if __name__ == "__main__":
    print("Meta-Reasoning Agent Wrappers for TextArena")
    print("=" * 60)
    print("\n1. MetaReasoningAgentWrapper - Coach + Player architecture")
    print("2. WorldModelAgentWrapper - Environment simulation & forecasting")
    print("\nSee example_meta_reasoning() and example_world_model() for usage.")
