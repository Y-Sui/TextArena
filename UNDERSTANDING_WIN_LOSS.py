"""
Understanding Win/Loss in Test-Time Scaling Evaluation
======================================================

This document explains what "win" and "loss" mean in the evaluation framework.
"""

# =============================================================================
# THE KEY CONCEPT
# =============================================================================

# Win/Loss means: Did the SCALING APPROACH beat the BASELINE opponent in that game?
#
# - WIN  = Scaling approach agent won the game
# - LOSS = Baseline opponent won the game
# - DRAW = Tie game

# =============================================================================
# HOW IT WORKS
# =============================================================================

"""
Step 1: Create two agents
--------------------------
Agent A: Uses test-time scaling (e.g., self-consistency with 5 samples)
Agent B: Baseline opponent (single forward pass, no scaling)

Step 2: Play multiple games
----------------------------
They play head-to-head in actual games (like TicTacToe, Chess, etc.)

Game 1: A is Player 0, B is Player 1  →  A wins  →  Count as WIN
Game 2: B is Player 0, A is Player 1  →  A wins  →  Count as WIN
Game 3: A is Player 0, B is Player 1  →  B wins  →  Count as LOSS
Game 4: B is Player 0, A is Player 1  →  Tie    →  Count as DRAW

Step 3: Calculate win rate
---------------------------
Win Rate = Wins / Total Games = 2/4 = 50%

This tells you: "The scaling approach won 50% of games against the baseline"
"""

# =============================================================================
# VISUAL EXAMPLE
# =============================================================================

def example_evaluation():
    """
    Visual walkthrough of what happens during evaluation
    """

    print("="*70)
    print("EXAMPLE: Evaluating 'self_consistency_5' on TicTacToe")
    print("="*70)

    print("\n1. SETUP")
    print("-" * 70)
    print("Agent A: Self-consistency with 5 samples (TEST-TIME SCALING)")
    print("  → Generates 5 different moves, picks most common")
    print("  → Slower but potentially better decisions")
    print()
    print("Agent B: Baseline (SINGLE FORWARD PASS)")
    print("  → Generates 1 move")
    print("  → Faster but may make worse decisions")

    print("\n2. PLAYING GAMES")
    print("-" * 70)

    # Game 1
    print("\nGame 1/5:")
    print("  A (self_consistency_5) plays as X")
    print("  B (baseline) plays as O")
    print("  → A makes better moves and wins!")
    print("  → Result: WIN for self_consistency_5")

    # Game 2
    print("\nGame 2/5:")
    print("  B (baseline) plays as X")
    print("  A (self_consistency_5) plays as O")
    print("  → A still wins even going second!")
    print("  → Result: WIN for self_consistency_5")

    # Game 3
    print("\nGame 3/5:")
    print("  A (self_consistency_5) plays as X")
    print("  B (baseline) plays as O")
    print("  → B gets lucky and wins")
    print("  → Result: LOSS for self_consistency_5")

    # Game 4
    print("\nGame 4/5:")
    print("  B (baseline) plays as X")
    print("  A (self_consistency_5) plays as O")
    print("  → Neither can win, game ends in draw")
    print("  → Result: DRAW")

    # Game 5
    print("\nGame 5/5:")
    print("  A (self_consistency_5) plays as X")
    print("  B (baseline) plays as O")
    print("  → A wins again!")
    print("  → Result: WIN for self_consistency_5")

    print("\n3. FINAL STATISTICS")
    print("-" * 70)
    print("Wins:  3")
    print("Draws: 1")
    print("Losses: 1")
    print("Win Rate: 3/5 = 60%")
    print()
    print("INTERPRETATION: Self-consistency won 60% of games against baseline.")
    print("This suggests the test-time scaling approach provides an advantage!")

# =============================================================================
# KEY POINTS
# =============================================================================

"""
Q: What does a high win rate mean?
A: The scaling approach is better than baseline at winning the game.
   Example: 80% win rate = scaling approach won 8 out of 10 games

Q: What does a 50% win rate mean?
A: The scaling approach is about the same as baseline (no improvement)

Q: What does a low win rate (<50%) mean?
A: The scaling approach is actually WORSE than baseline!
   (Maybe the extra complexity hurts performance, or it's too slow)

Q: Why alternate starting positions (Player 0 vs Player 1)?
A: Many games have first-move advantage. Alternating ensures fair comparison.
   - Game 1: Scaling approach goes first
   - Game 2: Baseline goes first
   - Game 3: Scaling approach goes first
   - etc.

Q: What's the difference between "baseline" approach and opponent?
A: They're the same! The "baseline" approach IS the opponent.
   - baseline approach = single forward pass, no test-time scaling
   - All other approaches are compared against this baseline

Q: Can I compare two scaling approaches directly?
A: Yes! You can modify the code to have any two agents play against each other.
   But the default setup compares each scaling approach vs baseline.
"""

# =============================================================================
# CODE WALKTHROUGH
# =============================================================================

"""
Here's what happens in the code:

1. evaluate_approach() is called with:
   - approach_name = "self_consistency_5"
   - agent = SelfConsistencyWrapper(base_agent, n_samples=5)
   - opponent = baseline agent (default)

2. For each game i from 0 to n_games:

   a) Alternate positions:
      if i % 2 == 0:
          agent1, agent2 = agent, opponent      # Scaling approach is Player 0
          player_id = 0
      else:
          agent1, agent2 = opponent, agent      # Scaling approach is Player 1
          player_id = 1

   b) Play the game:
      game_result = run_single_game(env_id, agent1, agent2, seed=seed)

   c) Check who won:
      if game_result["winner"] == player_id:    # Did scaling approach win?
          wins += 1
          print("WIN")
      elif game_result["winner"] is None:
          draws += 1
          print("DRAW")
      else:
          print("LOSS")                         # Baseline won

3. Calculate statistics:
   win_rate = wins / n_games

4. Return results showing how well scaling approach performed vs baseline
"""

# =============================================================================
# COMPARISON TABLE INTERPRETATION
# =============================================================================

"""
When you see a summary table like this:

TicTacToe-v0
--------------------------------------------------------------------------------
Approach                       Win Rate     Avg Time     Avg Turns
--------------------------------------------------------------------------------
self_consistency_5              80.0%       42.1s         8.5
iterative_refinement_2          75.0%       38.2s         8.3
best_of_3                       65.0%       28.5s         8.1
baseline                        50.0%        8.3s         7.9

This means:

- self_consistency_5 beat baseline in 80% of games (very good!)
- iterative_refinement_2 beat baseline in 75% of games (also good!)
- best_of_3 beat baseline in 65% of games (moderate improvement)
- baseline vs baseline = 50% (as expected, random chance when playing itself)

The "baseline" row shows 50% because when the framework evaluates baseline,
it's playing baseline vs baseline, which should be roughly 50-50.

TIME INTERPRETATION:
- self_consistency_5 takes 42.1s per game (5x slower due to 5 samples)
- baseline takes 8.3s per game
- Trade-off: 5x compute cost for +30% win rate improvement
"""

# =============================================================================
# PRACTICAL EXAMPLE
# =============================================================================

def real_world_interpretation():
    """
    What these results mean for practical applications
    """

    print("\n" + "="*70)
    print("REAL-WORLD INTERPRETATION")
    print("="*70)

    print("\nScenario: You want to deploy an LLM agent to play Chess")
    print()
    print("Option 1: Baseline (single forward pass)")
    print("  - Win rate: 45% vs human players")
    print("  - Response time: 2 seconds per move")
    print("  - Cost: $0.01 per game")
    print()
    print("Option 2: Self-consistency with 5 samples")
    print("  - Win rate: 65% vs human players  ← +20% improvement!")
    print("  - Response time: 10 seconds per move  ← 5x slower")
    print("  - Cost: $0.05 per game  ← 5x more expensive")
    print()
    print("Decision:")
    print("  - If winning is critical → Use self-consistency")
    print("  - If speed/cost matters → Use baseline")
    print("  - If balanced → Try best_of_3 (middle ground)")
    print()
    print("The evaluation framework helps you make this trade-off decision!")

# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    example_evaluation()
    real_world_interpretation()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
WIN  = Test-time scaling approach beat the baseline in that game
LOSS = Baseline beat the test-time scaling approach in that game
DRAW = Neither won (tie)

Win Rate = (Wins / Total Games) × 100%

High win rate (>60%) = Scaling approach is working well!
Medium win rate (50-60%) = Small improvement
Low win rate (<50%) = Scaling approach is actually worse than baseline
""")
