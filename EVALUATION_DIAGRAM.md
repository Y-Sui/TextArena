# Evaluation Flow Diagram

## How Win/Loss is Determined

```
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION SETUP                             │
└─────────────────────────────────────────────────────────────────┘

Agent A: Self-Consistency (5 samples)  ←─ Test-time scaling approach
    ↓
    ├─ Generate 5 moves
    ├─ Pick most common
    └─ Return final move

                    VS

Agent B: Baseline (1 sample)  ←─ Standard approach (opponent)
    ↓
    ├─ Generate 1 move
    └─ Return move


┌─────────────────────────────────────────────────────────────────┐
│                     GAME 1: A goes first                         │
└─────────────────────────────────────────────────────────────────┘

    TicTacToe Board:

    Agent A (Player 0, X)  vs  Agent B (Player 1, O)

    Turn 1: A plays [4] (center)     X | . | .
                                     ---------
    Turn 2: B plays [0] (top-left)   . | . | .
                                     ---------
    Turn 3: A plays [2]              . | . | .
    ... game continues ...

    Final Board:                     O | . | X
                                     ---------
                                     O | X | X    ← A wins!
                                     ---------
                                     . | . | X

    Result: Agent A wins → rewards = {0: 1, 1: -1}

    ✓ WIN counted for self_consistency_5

    Why? Because Agent A (self_consistency_5) = Player 0, and Player 0 won!


┌─────────────────────────────────────────────────────────────────┐
│                     GAME 2: B goes first                         │
└─────────────────────────────────────────────────────────────────┘

    TicTacToe Board:

    Agent B (Player 0, X)  vs  Agent A (Player 1, O)

    Turn 1: B plays [4] (center)     . | . | .
                                     ---------
    Turn 2: A plays [0] (top-left)   . | X | .
                                     ---------
    Turn 3: B plays [8]              . | . | .
    ... game continues ...

    Final Board:                     O | O | O    ← A wins!
                                     ---------
                                     X | X | .
                                     ---------
                                     . | . | X

    Result: Agent A wins → rewards = {0: -1, 1: 1}

    ✓ WIN counted for self_consistency_5

    Why? Because Agent A (self_consistency_5) = Player 1, and Player 1 won!


┌─────────────────────────────────────────────────────────────────┐
│                     GAME 3: A goes first                         │
└─────────────────────────────────────────────────────────────────┘

    Agent A (Player 0, X)  vs  Agent B (Player 1, O)

    Final Board:                     O | O | X
                                     ---------
                                     O | X | X
                                     ---------
                                     X | O | O    ← B wins!

    Result: Agent B wins → rewards = {0: -1, 1: 1}

    ✗ LOSS counted for self_consistency_5

    Why? Because Agent A (self_consistency_5) = Player 0, but Player 1 won!


┌─────────────────────────────────────────────────────────────────┐
│                     FINAL STATISTICS                             │
└─────────────────────────────────────────────────────────────────┘

Total games: 10
Wins: 7    ← self_consistency_5 won 7 games
Draws: 1   ← 1 game ended in tie
Losses: 2  ← baseline won 2 games

Win Rate = 7/10 = 70%

INTERPRETATION:
  Self-consistency with 5 samples beat the baseline in 70% of games!
  This is strong evidence that test-time scaling improves performance.
```

---

## Key Logic in Code

```python
# In evaluate_approach() method:

for i in range(n_games):
    # Alternate who goes first
    if i % 2 == 0:
        agent1, agent2 = scaling_agent, baseline
        player_id = 0  # Scaling agent is Player 0
    else:
        agent1, agent2 = baseline, scaling_agent
        player_id = 1  # Scaling agent is Player 1

    # Play the game
    game_result = run_single_game(env_id, agent1, agent2, seed)

    # Check if scaling agent won
    if game_result["winner"] == player_id:
        wins += 1  # Scaling agent won!
    elif game_result["winner"] is None:
        draws += 1  # Tie
    else:
        # Baseline won (scaling agent lost)
        pass

# Calculate performance
win_rate = wins / n_games
```

---

## What Are We Measuring?

### ❌ We are NOT measuring:
- How well the agent plays in absolute terms
- Whether the agent beats humans
- Whether the agent plays perfectly

### ✅ We ARE measuring:
- **Does test-time scaling improve performance vs baseline?**
- By how much? (win rate)
- At what cost? (time, compute)

---

## Example Comparison Table

```
TicTacToe-v0 (10 games per approach)
─────────────────────────────────────────────────────
Approach              Win Rate    Compute Cost
─────────────────────────────────────────────────────
self_consistency_5      70%         5x
best_of_3              65%         4x
iterative_refine_1     60%         3x
baseline               50%         1x
─────────────────────────────────────────────────────
```

**Interpretation:**
- **self_consistency_5** won 70% of games against baseline
  - Trade-off: 5x compute cost for +20% win rate

- **best_of_3** won 65% of games against baseline
  - Trade-off: 4x compute cost for +15% win rate

- **baseline** vs baseline = 50%
  - When baseline plays itself, it wins ~50% (as expected)

---

## Real-World Analogy

Think of it like testing different chess strategies:

1. **Baseline Strategy**: Think for 10 seconds per move

2. **Test-Time Scaling Strategy**: Think for 50 seconds per move
   - Consider 5 different moves
   - Evaluate which is best
   - Pick the best one

Now we play 10 games:
- Strategy 2 wins 7 games
- Strategy 1 wins 2 games
- 1 draw

**Conclusion**: Spending 5x more time thinking improves your chess performance by 20%!

The evaluation framework does exactly this, but for LLM agents playing various games.

---

## Bottom Line

**WIN** = Your test-time scaling approach beat a baseline agent in a head-to-head game

**High win rate (>60%)** = Your scaling approach is effective! 🎉

**Low win rate (<50%)** = Your scaling approach actually hurts performance 😞

**~50% win rate** = No improvement, just added cost 😐

The framework helps you discover which test-time scaling techniques actually work for different game types!
