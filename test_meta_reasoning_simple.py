"""
Simple test runner for Meta-Reasoning Agent Wrappers (no pytest required)
"""

import os
import sys

# Add textarena to path
sys.path.insert(0, os.path.dirname(__file__))

import textarena as ta
from textarena.agents.meta_reasoning_wrappers import (
    MetaReasoningAgentWrapper,
    WorldModelAgentWrapper
)


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
# TEST FUNCTIONS
# ============================================================================

def test_meta_reasoning_initialization():
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
    print("✓ test_meta_reasoning_initialization passed")


def test_coach_provides_guidance():
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

    assert coach.call_count == 1
    assert player.call_count == 1
    assert action == "[move e4]"
    print("✓ test_coach_provides_guidance passed")


def test_player_receives_guidance():
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

    assert "COACH'S STRATEGIC GUIDANCE" in player.last_observation
    assert "controlling the center" in player.last_observation.lower()
    print("✓ test_player_receives_guidance passed")


def test_interaction_history_tracking():
    """Test that interaction history is properly tracked"""
    player = MockAgent(responses=["[action 1]", "[action 2]", "[action 3]"])
    coach = MockCoachAgent()

    meta_agent = MetaReasoningAgentWrapper(
        agent=player,
        coach_agent=coach
    )

    for i in range(3):
        meta_agent(f"observation {i}")

    assert len(meta_agent.interaction_history) == 3
    print("✓ test_interaction_history_tracking passed")


def test_meta_reasoning_reset_history():
    """Test reset_history method"""
    player = MockAgent()
    coach = MockCoachAgent()

    meta_agent = MetaReasoningAgentWrapper(
        agent=player,
        coach_agent=coach
    )

    meta_agent("observation 1")
    meta_agent("observation 2")
    assert len(meta_agent.interaction_history) == 2

    meta_agent.reset_history()
    assert len(meta_agent.interaction_history) == 0
    print("✓ test_meta_reasoning_reset_history passed")


def test_world_model_initialization():
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
    print("✓ test_world_model_initialization passed")


def test_candidate_generation():
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

    assert agent.call_count >= 3
    print("✓ test_candidate_generation passed")


def test_action_simulation():
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

    assert simulator.call_count == 3
    print("✓ test_action_simulation passed")


def test_simulation_caching():
    """Test that simulation results are cached"""
    agent = MockAgent(responses=["[same action]"] * 6)
    simulator = MockSimulatorAgent()

    world_model = WorldModelAgentWrapper(
        agent=agent,
        simulator_agent=simulator,
        n_candidates=3,
        debugging=False
    )

    world_model("observation")
    world_model("observation")

    assert len(world_model.simulation_cache) > 0
    print("✓ test_simulation_caching passed")


def test_action_history_tracking():
    """Test that action history is properly tracked"""
    agent = MockAgent(responses=["[action 1]", "[action 2]"])
    simulator = MockSimulatorAgent()

    world_model = WorldModelAgentWrapper(
        agent=agent,
        simulator_agent=simulator,
        n_candidates=2,
        use_action_history=True
    )

    world_model("observation 1")
    world_model("observation 2")

    assert len(world_model.action_history) == 2
    print("✓ test_action_history_tracking passed")


def test_world_model_reset_history():
    """Test reset_history method"""
    agent = MockAgent()
    simulator = MockSimulatorAgent()

    world_model = WorldModelAgentWrapper(
        agent=agent,
        simulator_agent=simulator
    )

    world_model("observation 1")
    world_model("observation 2")

    assert len(world_model.action_history) > 0
    assert len(world_model.simulation_cache) > 0

    world_model.reset_history()

    assert len(world_model.action_history) == 0
    assert len(world_model.simulation_cache) == 0
    print("✓ test_world_model_reset_history passed")


def test_learned_patterns():
    """Test get_learned_patterns method"""
    agent = MockAgent(responses=["[action]"])
    simulator = MockSimulatorAgent(scores=[8.0, 3.0, 7.5])

    world_model = WorldModelAgentWrapper(
        agent=agent,
        simulator_agent=simulator,
        n_candidates=1
    )

    world_model("observation 1")
    world_model("observation 2")
    world_model("observation 3")

    patterns = world_model.get_learned_patterns()

    assert 'total_decisions' in patterns
    assert patterns['total_decisions'] == 3
    assert 'successful_actions' in patterns
    assert 'unsuccessful_actions' in patterns
    assert 'average_score' in patterns
    print("✓ test_learned_patterns passed")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all test functions"""
    print("=" * 70)
    print("Running Meta-Reasoning Agent Wrapper Tests")
    print("=" * 70)
    print()

    # Meta-Reasoning tests
    print("Testing MetaReasoningAgentWrapper...")
    print("-" * 70)
    test_meta_reasoning_initialization()
    test_coach_provides_guidance()
    test_player_receives_guidance()
    test_interaction_history_tracking()
    test_meta_reasoning_reset_history()
    print()

    # World Model tests
    print("Testing WorldModelAgentWrapper...")
    print("-" * 70)
    test_world_model_initialization()
    test_candidate_generation()
    test_action_simulation()
    test_simulation_caching()
    test_action_history_tracking()
    test_world_model_reset_history()
    test_learned_patterns()
    print()

    print("=" * 70)
    print("All tests passed successfully! ✓")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
