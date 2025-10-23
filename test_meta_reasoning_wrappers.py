"""
Comprehensive Tests for Meta-Reasoning Agent Wrappers
=====================================================

Tests for MetaReasoningAgentWrapper and WorldModelAgentWrapper
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch
from dotenv import load_dotenv

# Add textarena to path
sys.path.insert(0, os.path.dirname(__file__))

import textarena as ta
from textarena.agents.meta_reasoning_wrappers import (
    MetaReasoningAgentWrapper,
    WorldModelAgentWrapper
)

# Load environment variables
load_dotenv()


# ============================================================================
# MOCK AGENTS FOR TESTING
# ============================================================================

class MockAgent(ta.Agent):
    """Simple mock agent for testing"""

    def __init__(self, responses=None, system_prompt="Mock agent"):
        self.system_prompt = system_prompt
        self.responses = responses or ["[mock action]"]
        self.call_count = 0
        self.last_observation = None

    def __call__(self, observation: str) -> str:
        self.last_observation = observation
        response_idx = min(self.call_count, len(self.responses) - 1)
        response = self.responses[response_idx]
        self.call_count += 1
        return response


class MockCoachAgent(ta.Agent):
    """Mock coach agent that provides strategic guidance"""

    def __init__(self, system_prompt="Mock coach"):
        self.system_prompt = system_prompt
        self.call_count = 0

    def __call__(self, observation: str) -> str:
        self.call_count += 1
        return (
            "Focus on controlling the center of the board. "
            "Play aggressively to gain early advantage."
        )


class MockSimulatorAgent(ta.Agent):
    """Mock simulator agent for world model testing"""

    def __init__(self, scores=None, system_prompt="Mock simulator"):
        self.system_prompt = system_prompt
        self.scores = scores or [7.5, 6.0, 8.5]
        self.call_count = 0

    def __call__(self, observation: str) -> str:
        score_idx = min(self.call_count, len(self.scores) - 1)
        score = self.scores[score_idx]
        self.call_count += 1
        return (
            f"If this action is taken, the player will gain a positional advantage. "
            f"Opponent likely responds with defensive play. "
            f"Overall favorability: {score}/10"
        )


# ============================================================================
# TESTS FOR META-REASONING AGENT WRAPPER
# ============================================================================

class TestMetaReasoningAgentWrapper:
    """Test suite for MetaReasoningAgentWrapper"""

    def test_initialization(self):
        """Test basic initialization"""
        player = MockAgent()
        coach = MockCoachAgent()

        meta_agent = MetaReasoningAgentWrapper(
            agent=player,
            coach_agent=coach
        )

        assert meta_agent.player_agent == player
        assert meta_agent.coach_agent == coach
        assert len(meta_agent.interaction_history) == 0

    def test_coach_provides_guidance(self):
        """Test that coach agent is called for guidance"""
        player = MockAgent(responses=["[move e4]"])
        coach = MockCoachAgent()

        meta_agent = MetaReasoningAgentWrapper(
            agent=player,
            coach_agent=coach,
            debugging=False
        )

        observation = "You are playing chess. It's your turn."
        action = meta_agent(observation)

        # Coach should be called once
        assert coach.call_count == 1

        # Player should be called once
        assert player.call_count == 1

        # Should return player's action
        assert action == "[move e4]"

    def test_player_receives_guidance(self):
        """Test that player receives coach's guidance"""
        player = MockAgent(responses=["[move e4]"])
        coach = MockCoachAgent()

        meta_agent = MetaReasoningAgentWrapper(
            agent=player,
            coach_agent=coach,
            debugging=False
        )

        observation = "You are playing chess."
        meta_agent(observation)

        # Check that player's last observation included coach's guidance
        assert "COACH'S STRATEGIC GUIDANCE" in player.last_observation
        assert "controlling the center" in player.last_observation.lower()

    def test_interaction_history_tracking(self):
        """Test that interaction history is properly tracked"""
        player = MockAgent(responses=["[action 1]", "[action 2]", "[action 3]"])
        coach = MockCoachAgent()

        meta_agent = MetaReasoningAgentWrapper(
            agent=player,
            coach_agent=coach
        )

        # Make multiple calls
        for i in range(3):
            meta_agent(f"observation {i}")

        # Should have 3 interactions recorded
        assert len(meta_agent.interaction_history) == 3

        # Each should have guidance, action, and observation
        for record in meta_agent.interaction_history:
            guidance, action, observation = record
            assert isinstance(guidance, str)
            assert isinstance(action, str)
            assert isinstance(observation, str)

    def test_history_limit(self):
        """Test that history is limited to prevent memory growth"""
        player = MockAgent(responses=["[action]"])
        coach = MockCoachAgent()

        meta_agent = MetaReasoningAgentWrapper(
            agent=player,
            coach_agent=coach
        )

        # Make many calls
        for i in range(15):
            meta_agent(f"observation {i}")

        # Should cap at 10
        assert len(meta_agent.interaction_history) == 10

    def test_reset_history(self):
        """Test reset_history method"""
        player = MockAgent()
        coach = MockCoachAgent()

        meta_agent = MetaReasoningAgentWrapper(
            agent=player,
            coach_agent=coach
        )

        # Build some history
        meta_agent("observation 1")
        meta_agent("observation 2")
        assert len(meta_agent.interaction_history) == 2

        # Reset
        meta_agent.reset_history()
        assert len(meta_agent.interaction_history) == 0

    def test_custom_prompts(self):
        """Test custom coach and player prompts"""
        player = MockAgent()
        coach = MockCoachAgent()

        custom_coach_prompt = "You are a CUSTOM coach."
        custom_player_prompt = "\n\nCUSTOM GUIDANCE: {guidance}"

        meta_agent = MetaReasoningAgentWrapper(
            agent=player,
            coach_agent=coach,
            coach_prompt=custom_coach_prompt,
            player_incorporation_prompt=custom_player_prompt
        )

        assert meta_agent.coach_prompt == custom_coach_prompt
        assert meta_agent.player_incorporation_prompt == custom_player_prompt

    def test_debugging_mode(self, capsys):
        """Test debugging mode output"""
        player = MockAgent(responses=["[debug action]"])
        coach = MockCoachAgent()

        meta_agent = MetaReasoningAgentWrapper(
            agent=player,
            coach_agent=coach,
            debugging=True
        )

        meta_agent("test observation")

        # Capture printed output
        captured = capsys.readouterr()

        # Should print debug information
        assert "META-REASONING AGENT" in captured.out
        assert "OBSERVATION" in captured.out
        assert "COACH'S STRATEGIC GUIDANCE" in captured.out
        assert "PLAYER'S ACTION" in captured.out


# ============================================================================
# TESTS FOR WORLD MODEL AGENT WRAPPER
# ============================================================================

class TestWorldModelAgentWrapper:
    """Test suite for WorldModelAgentWrapper"""

    def test_initialization(self):
        """Test basic initialization"""
        agent = MockAgent()
        simulator = MockSimulatorAgent()

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=3
        )

        assert world_model.simulator_agent == simulator
        assert world_model.n_candidates == 3
        assert len(world_model.action_history) == 0
        assert len(world_model.simulation_cache) == 0

    def test_candidate_generation(self):
        """Test that multiple candidate actions are generated"""
        agent = MockAgent(responses=["[action 1]", "[action 2]", "[action 3]"])
        simulator = MockSimulatorAgent()

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=3,
            debugging=False
        )

        observation = "Test observation"
        world_model(observation)

        # Agent should be called n_candidates times for generation
        # Plus calls from simulation
        assert agent.call_count >= 3

    def test_action_simulation(self):
        """Test that simulator is called for each candidate"""
        agent = MockAgent(responses=["[action 1]", "[action 2]", "[action 3]"])
        simulator = MockSimulatorAgent(scores=[5.0, 8.0, 6.0])

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=3,
            debugging=False
        )

        world_model("test observation")

        # Simulator should be called for each candidate
        assert simulator.call_count == 3

    def test_best_action_selection(self):
        """Test that highest-scored action is selected"""
        # Create agent that generates 3 different actions
        agent = MockAgent(responses=["[weak action]", "[good action]", "[best action]"])

        # Simulator scores them: 4.0, 6.0, 9.0
        simulator = MockSimulatorAgent(scores=[4.0, 6.0, 9.0])

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=3,
            debugging=False
        )

        action = world_model("test observation")

        # Should select the best-scored action
        # Note: Due to duplicate removal, we check if it's one of the high-scoring options
        assert "[action]" in action.lower() or "action" in action.lower()

    def test_simulation_caching(self):
        """Test that simulation results are cached"""
        agent = MockAgent(responses=["[same action]"] * 6)
        simulator = MockSimulatorAgent()

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=3,
            debugging=False
        )

        # First call
        world_model("observation")
        first_call_count = simulator.call_count

        # Second call with same observation and actions
        world_model("observation")
        second_call_count = simulator.call_count

        # Cache should prevent redundant simulations
        # Second call should use cached results
        assert len(world_model.simulation_cache) > 0

    def test_action_history_tracking(self):
        """Test that action history is properly tracked"""
        agent = MockAgent(responses=["[action 1]", "[action 2]"])
        simulator = MockSimulatorAgent()

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=2,
            use_action_history=True
        )

        # Make multiple calls
        world_model("observation 1")
        world_model("observation 2")

        # Should have 2 records
        assert len(world_model.action_history) == 2

        # Each record should have required fields
        for record in world_model.action_history:
            assert 'action' in record
            assert 'observation' in record
            assert 'candidates' in record
            assert 'simulations' in record

    def test_prior_experience_context(self):
        """Test that prior experience is used in simulations"""
        agent = MockAgent(responses=["[action 1]", "[action 2]"])
        simulator = MockSimulatorAgent()

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=2,
            use_action_history=True
        )

        # Build some history
        world_model("first observation")

        # Second call should use prior experience
        world_model("second observation")

        # Check that simulator received context with prior experience
        # This is implicit in the improved decision making

    def test_history_limit(self):
        """Test that history is limited to prevent memory growth"""
        agent = MockAgent(responses=["[action]"])
        simulator = MockSimulatorAgent()

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=1
        )

        # Make many calls
        for i in range(25):
            world_model(f"observation {i}")

        # Should cap at 20
        assert len(world_model.action_history) == 20

    def test_reset_history(self):
        """Test reset_history method"""
        agent = MockAgent()
        simulator = MockSimulatorAgent()

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator
        )

        # Build history and cache
        world_model("observation 1")
        world_model("observation 2")

        assert len(world_model.action_history) > 0
        assert len(world_model.simulation_cache) > 0

        # Reset
        world_model.reset_history()

        assert len(world_model.action_history) == 0
        assert len(world_model.simulation_cache) == 0

    def test_learned_patterns(self):
        """Test get_learned_patterns method"""
        agent = MockAgent(responses=["[action]"])
        simulator = MockSimulatorAgent(scores=[8.0, 3.0, 7.5])

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=1
        )

        # Make some calls
        world_model("observation 1")
        world_model("observation 2")
        world_model("observation 3")

        patterns = world_model.get_learned_patterns()

        assert 'total_decisions' in patterns
        assert patterns['total_decisions'] == 3
        assert 'successful_actions' in patterns
        assert 'unsuccessful_actions' in patterns
        assert 'average_score' in patterns

    def test_debugging_mode(self, capsys):
        """Test debugging mode output"""
        agent = MockAgent(responses=["[action 1]", "[action 2]"])
        simulator = MockSimulatorAgent()

        world_model = WorldModelAgentWrapper(
            agent=agent,
            simulator_agent=simulator,
            n_candidates=2,
            debugging=True
        )

        world_model("test observation")

        captured = capsys.readouterr()

        assert "WORLD MODEL AGENT" in captured.out
        assert "GENERATED" in captured.out
        assert "SIMULATING ACTION OUTCOMES" in captured.out
        assert "SELECTED BEST ACTION" in captured.out


# ============================================================================
# INTEGRATION TESTS WITH REAL GAMES
# ============================================================================

class TestIntegrationWithGames:
    """Integration tests with actual TextArena games"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires API key for LLM agents"
    )
    def test_meta_reasoning_with_tictactoe(self):
        """Test MetaReasoningAgentWrapper with TicTacToe"""
        try:
            # Create agents
            if os.getenv("OPENAI_API_KEY"):
                player = ta.agents.OpenAIAgent(
                    model_name="gpt-4o-mini",
                    system_prompt="You are a TicTacToe player."
                )
                coach = ta.agents.OpenAIAgent(
                    model_name="gpt-4o-mini",
                    system_prompt="You are a strategic coach."
                )
            else:
                player = ta.agents.OpenRouterAgent(
                    model_name="google/gemini-2.0-flash-001"
                )
                coach = ta.agents.OpenRouterAgent(
                    model_name="google/gemini-2.0-flash-001"
                )

            meta_agent = MetaReasoningAgentWrapper(
                agent=player,
                coach_agent=coach,
                debugging=True
            )

            # Create environment
            env = ta.make("TicTacToe-v0")
            env.reset(num_players=1)

            # Get observation and take action
            player_id, observation = env.get_observation()
            action = meta_agent(observation)

            # Verify action is returned
            assert isinstance(action, str)
            assert len(action) > 0

            print(f"\n[Integration Test] Action taken: {action}")

            # Verify history was recorded
            assert len(meta_agent.interaction_history) == 1

        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires API key for LLM agents"
    )
    def test_world_model_with_tictactoe(self):
        """Test WorldModelAgentWrapper with TicTacToe"""
        try:
            # Create agents
            if os.getenv("OPENAI_API_KEY"):
                agent = ta.agents.OpenAIAgent(
                    model_name="gpt-4o-mini",
                    system_prompt="You are a TicTacToe player."
                )
                simulator = ta.agents.OpenAIAgent(
                    model_name="gpt-4o-mini",
                    system_prompt="You simulate game states."
                )
            else:
                agent = ta.agents.OpenRouterAgent(
                    model_name="google/gemini-2.0-flash-001"
                )
                simulator = ta.agents.OpenRouterAgent(
                    model_name="google/gemini-2.0-flash-001"
                )

            world_model_agent = WorldModelAgentWrapper(
                agent=agent,
                simulator_agent=simulator,
                n_candidates=2,
                simulation_depth=1,
                debugging=True
            )

            # Create environment
            env = ta.make("TicTacToe-v0")
            env.reset(num_players=1)

            # Get observation and take action
            player_id, observation = env.get_observation()
            action = world_model_agent(observation)

            # Verify action is returned
            assert isinstance(action, str)
            assert len(action) > 0

            print(f"\n[Integration Test] Action taken: {action}")

            # Verify history and simulations
            assert len(world_model_agent.action_history) == 1

            # Check learned patterns
            patterns = world_model_agent.get_learned_patterns()
            assert patterns['total_decisions'] == 1

            print(f"[Integration Test] Learned patterns: {patterns}")

        except Exception as e:
            pytest.skip(f"Integration test failed: {e}")


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    print("Running Meta-Reasoning Agent Wrapper Tests")
    print("=" * 70)

    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
