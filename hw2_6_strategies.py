import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Multi-Armed Bandit Homework Script
# 6 strategies: A/B Test, Optimistic, Epsilon-Greedy,
#               Softmax, UCB, Thompson Sampling
# Problem setup follows the assignment wording:
#   - Arms: A, B, C
#   - Means: A=0.8, B=0.7, C=0.5
#   - Budget = $10,000  -> treat as 10,000 pulls, $1 each
# Outputs:
#   1) class comparison table
#   2) average allocation plan across methods
#   3) cumulative reward plot (smoothed by many runs)
#   4) cumulative expected regret plot (smoothed)
# ============================================================

SEED = 42
rng_global = np.random.default_rng(SEED)

ARMS = ["A", "B", "C"]
TRUE_MEANS = np.array([0.8, 0.7, 0.5], dtype=float)
N_ARMS = len(ARMS)
T = 10_000
N_RUNS = 200   # increase for smoother curves
OPTIMAL_ARM = int(np.argmax(TRUE_MEANS))
OPTIMAL_MEAN = float(np.max(TRUE_MEANS))
BUDGET = T


# ---------------------- core helpers ----------------------
def pull(arm: int, rng: np.random.Generator) -> int:
    return int(rng.random() < TRUE_MEANS[arm])


def summarize_single_run(chosen_arms: np.ndarray, rewards: np.ndarray):
    counts = np.bincount(chosen_arms, minlength=N_ARMS)
    alloc = counts.astype(float)
    expected_total_reward = float(np.sum(counts * TRUE_MEANS))
    realized_total_reward = float(np.sum(rewards))
    regret = T * OPTIMAL_MEAN - expected_total_reward
    return alloc, expected_total_reward, realized_total_reward, regret


# ---------------------- strategies ------------------------
def run_ab_test(rng: np.random.Generator):
    chosen = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=int)

    # Static exploration plan: A/B/C evenly during the first 3,000 pulls
    warmup = 3000
    for t in range(warmup):
        arm = t % N_ARMS
        chosen[t] = arm
        rewards[t] = pull(arm, rng)

    empirical_means = np.array([
        rewards[chosen == arm].mean() if np.any(chosen[:warmup] == arm) else 0.0
        for arm in range(N_ARMS)
    ])
    best_arm = int(np.argmax(empirical_means))

    for t in range(warmup, T):
        chosen[t] = best_arm
        rewards[t] = pull(best_arm, rng)

    return chosen, rewards


def run_optimistic(rng: np.random.Generator, init_value: float = 1.0):
    q = np.ones(N_ARMS) * init_value
    n = np.zeros(N_ARMS)
    chosen = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=int)

    for t in range(T):
        arm = int(np.argmax(q))
        r = pull(arm, rng)
        chosen[t] = arm
        rewards[t] = r
        n[arm] += 1
        q[arm] += (r - q[arm]) / n[arm]

    return chosen, rewards


def run_epsilon_greedy(rng: np.random.Generator, eps: float = 0.1):
    q = np.zeros(N_ARMS)
    n = np.zeros(N_ARMS)
    chosen = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=int)

    for t in range(T):
        if rng.random() < eps:
            arm = int(rng.integers(N_ARMS))
        else:
            arm = int(np.argmax(q))
        r = pull(arm, rng)
        chosen[t] = arm
        rewards[t] = r
        n[arm] += 1
        q[arm] += (r - q[arm]) / n[arm]

    return chosen, rewards


def run_softmax(rng: np.random.Generator, temp: float = 0.1):
    q = np.zeros(N_ARMS)
    n = np.zeros(N_ARMS)
    chosen = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=int)

    for t in range(T):
        logits = q / temp
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        arm = int(rng.choice(N_ARMS, p=probs))
        r = pull(arm, rng)
        chosen[t] = arm
        rewards[t] = r
        n[arm] += 1
        q[arm] += (r - q[arm]) / n[arm]

    return chosen, rewards


def run_ucb(rng: np.random.Generator, c: float = 2.0):
    q = np.zeros(N_ARMS)
    n = np.zeros(N_ARMS)
    chosen = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=int)

    for t in range(T):
        if t < N_ARMS:
            arm = t
        else:
            bonus = c * np.sqrt(np.log(t + 1) / (n + 1e-9))
            arm = int(np.argmax(q + bonus))
        r = pull(arm, rng)
        chosen[t] = arm
        rewards[t] = r
        n[arm] += 1
        q[arm] += (r - q[arm]) / n[arm]

    return chosen, rewards


def run_thompson(rng: np.random.Generator):
    alpha = np.ones(N_ARMS)
    beta = np.ones(N_ARMS)
    chosen = np.zeros(T, dtype=int)
    rewards = np.zeros(T, dtype=int)

    for t in range(T):
        samples = rng.beta(alpha, beta)
        arm = int(np.argmax(samples))
        r = pull(arm, rng)
        chosen[t] = arm
        rewards[t] = r
        alpha[arm] += r
        beta[arm] += 1 - r

    return chosen, rewards


METHODS = {
    "A/B Test": run_ab_test,
    "Optimistic": run_optimistic,
    "Epsilon-Greedy": run_epsilon_greedy,
    "Softmax": run_softmax,
    "UCB": run_ucb,
    "Thompson": run_thompson,
}

EXPLORATION_STYLE = {
    "A/B Test": "Static",
    "Optimistic": "Implicit",
    "Epsilon-Greedy": "Random",
    "Softmax": "Probabilistic",
    "UCB": "Confidence-based",
    "Thompson": "Bayesian",
}

NOTES = {
    "A/B Test": "Simple but wasteful",
    "Optimistic": "Front-loaded exploration",
    "Epsilon-Greedy": "Easy baseline",
    "Softmax": "Smooth control",
    "UCB": "Efficient",
    "Thompson": "Best practical",
}

HOW_IT_WORKS = {
    "A/B Test": {
        "explore": "Uses a fixed early testing stage: splits the first 3,000 pulls evenly across A, B, C.",
        "exploit": "After the test phase, commits to the arm with the highest observed mean.",
    },
    "Optimistic": {
        "explore": "Starts every arm with an overly high initial estimate, so under-tried arms keep looking attractive.",
        "exploit": "As estimates correct downward, it settles on the arm with the strongest observed reward.",
    },
    "Epsilon-Greedy": {
        "explore": "With probability epsilon=0.1, picks a random arm.",
        "exploit": "With probability 0.9, chooses the current best estimated arm.",
    },
    "Softmax": {
        "explore": "Samples all arms with probabilities based on their estimated values.",
        "exploit": "Better arms get much higher probability, especially when one estimate becomes clearly best.",
    },
    "UCB": {
        "explore": "Adds an uncertainty bonus to less-tried arms, forcing targeted exploration.",
        "exploit": "Chooses the arm with the best value-plus-confidence score once uncertainty shrinks.",
    },
    "Thompson": {
        "explore": "Draws a random plausible reward rate from each arm's posterior, so uncertain arms are explored naturally.",
        "exploit": "Arms with stronger evidence produce stronger posterior samples and get chosen more often.",
    },
}


# ---------------------- simulation ------------------------
def simulate_method(name, fn, n_runs=N_RUNS):
    reward_curves = []
    regret_curves = []
    allocs = []
    expected_totals = []
    realized_totals = []
    regrets = []

    for run_idx in range(n_runs):
        rng = np.random.default_rng(SEED + run_idx * 1009 + hash(name) % 997)
        chosen, rewards = fn(rng)
        counts = np.bincount(chosen, minlength=N_ARMS)

        reward_curves.append(np.cumsum(rewards))
        instant_expected_regret = OPTIMAL_MEAN - TRUE_MEANS[chosen]
        regret_curves.append(np.cumsum(instant_expected_regret))

        alloc, expected_total, realized_total, regret = summarize_single_run(chosen, rewards)
        allocs.append(alloc)
        expected_totals.append(expected_total)
        realized_totals.append(realized_total)
        regrets.append(regret)

    return {
        "avg_cum_reward": np.mean(reward_curves, axis=0),
        "avg_cum_regret": np.mean(regret_curves, axis=0),
        "avg_alloc": np.mean(allocs, axis=0),
        "avg_expected_reward": float(np.mean(expected_totals)),
        "avg_realized_reward": float(np.mean(realized_totals)),
        "avg_regret": float(np.mean(regrets)),
    }


all_results = {name: simulate_method(name, fn) for name, fn in METHODS.items()}


# ---------------------- output tables ---------------------
rows = []
for name, res in all_results.items():
    alloc = res["avg_alloc"]
    rows.append({
        "Method": name,
        "Exploration Style": EXPLORATION_STYLE[name],
        "Allocate A ($)": round(float(alloc[0]), 0),
        "Allocate B ($)": round(float(alloc[1]), 0),
        "Allocate C ($)": round(float(alloc[2]), 0),
        "Total Expected Reward": round(res["avg_expected_reward"], 2),
        "Regret": round(res["avg_regret"], 2),
        "Notes": NOTES[name],
    })

comparison_df = pd.DataFrame(rows)
comparison_df = comparison_df.sort_values(by="Total Expected Reward", ascending=False).reset_index(drop=True)
comparison_df.to_csv("hw2_comparison_table.csv", index=False)

print("\n=== Class Comparison Table ===")
print(comparison_df.to_string(index=False))

print("\n=== Strategy Design Notes ===")
for name in METHODS:
    print(f"\n{name}")
    print(f"  Explore: {HOW_IT_WORKS[name]['explore']}")
    print(f"  Exploit: {HOW_IT_WORKS[name]['exploit']}")

print("\n=== Optimal Benchmark ===")
print(f"Always choose A -> expected total reward = {T * TRUE_MEANS[0]:.2f}")


# ---------------------- plots -----------------------------
plt.figure(figsize=(9, 6))
for name, res in all_results.items():
    plt.plot(res["avg_cum_reward"], label=name)
plt.title(f"Average Cumulative Reward ({N_RUNS} runs)")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.legend()
plt.tight_layout()
plt.savefig("hw2_cumulative_reward.png", dpi=160)
plt.close()

plt.figure(figsize=(9, 6))
for name, res in all_results.items():
    plt.plot(res["avg_cum_regret"], label=name)
plt.title(f"Average Cumulative Expected Regret ({N_RUNS} runs)")
plt.xlabel("Steps")
plt.ylabel("Regret")
plt.legend()
plt.tight_layout()
plt.savefig("hw2_cumulative_regret.png", dpi=160)
plt.close()

plt.figure(figsize=(9, 6))
for name, res in all_results.items():
    plt.plot(res["avg_cum_regret"], label=name)
plt.yscale("log")
plt.title(f"Average Cumulative Expected Regret (log scale, {N_RUNS} runs)")
plt.xlabel("Steps")
plt.ylabel("Regret")
plt.legend()
plt.tight_layout()
plt.savefig("hw2_cumulative_regret_log.png", dpi=160)
plt.close()

print("\nSaved files:")
print("- hw2_6_strategies.py")
print("- hw2_comparison_table.csv")
print("- hw2_cumulative_reward.png")
print("- hw2_cumulative_regret.png")
print("- hw2_cumulative_regret_log.png")
