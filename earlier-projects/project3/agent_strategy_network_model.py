import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# AGENT-BASED NETWORK MODEL
# Cooperate, Compete, or Stay Neutral
# ============================================================


# -----------------------------
# 1. PARAMETERS
# -----------------------------

RANDOM_SEED = 42

N = 30                      # Number of agents
C = 60                      # Number of edges/connections
STEPS = 100                 # Number of simulation steps

STARTING_ENERGY = 10
ENVIRONMENT_INCOME = 1      # Energy received from the environment each step
METABOLISM_COST = 1         # Energy required to survive each step

HELP_AMOUNT = 1             # Energy transferred by helpers
STEAL_AMOUNT = 1            # Energy taken by aggressive agents

STRATEGIES = ["helper", "neutral", "aggressive"]

OBSERVATION_STEPS = [0, 25, 50, 100]


# Make the simulation reproducible
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# -----------------------------
# 2. CREATE THE ENVIRONMENT
# -----------------------------

def create_connected_random_graph(number_of_agents, number_of_edges):
    """
    Creates a connected random graph.

    Nodes represent agents.
    Edges represent relationships that allow local interaction.
    """

    minimum_edges = number_of_agents - 1
    maximum_edges = number_of_agents * (number_of_agents - 1) // 2

    if number_of_edges < minimum_edges:
        raise ValueError(
            f"C must be at least {minimum_edges} for the graph to be connected."
        )

    if number_of_edges > maximum_edges:
        raise ValueError(
            f"C cannot be greater than {maximum_edges} for N={number_of_agents}."
        )

    while True:
        graph = nx.gnm_random_graph(
            number_of_agents,
            number_of_edges,
            seed=random.randint(0, 1_000_000)
        )

        if nx.is_connected(graph):
            return graph


# -----------------------------
# 3. INITIALIZE AGENTS
# -----------------------------

def initialize_agents(graph):
    """
    Gives agents balanced starting strategies and equal starting energy.

    With N = 30, this creates:
    10 helpers
    10 neutral agents
    10 aggressive agents
    """

    strategy_assignments = []

    base_amount = len(graph.nodes()) // len(STRATEGIES)
    remainder = len(graph.nodes()) % len(STRATEGIES)

    for index, strategy in enumerate(STRATEGIES):
        amount = base_amount

        if index < remainder:
            amount += 1

        strategy_assignments.extend([strategy] * amount)

    random.shuffle(strategy_assignments)

    for node, strategy in zip(graph.nodes(), strategy_assignments):
        graph.nodes[node]["energy"] = STARTING_ENERGY
        graph.nodes[node]["strategy"] = strategy
        graph.nodes[node]["alive"] = True


# -----------------------------
# 4. MEASUREMENT FUNCTIONS
# -----------------------------

def gini(values):
    """
    Calculates the Gini index.

    0 means complete equality.
    Larger values mean greater inequality.
    """

    values = np.array(values, dtype=float)

    if len(values) == 0:
        return 0.0

    if np.min(values) < 0:
        values = values - np.min(values)

    total = np.sum(values)

    if total == 0:
        return 0.0

    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)

    return (
        (2 * np.sum(index * values)) / (n * total)
        - (n + 1) / n
    )


def count_alive_by_strategy(graph):
    """
    Counts living agents for each strategy.
    """

    counts = {
        "helper": 0,
        "neutral": 0,
        "aggressive": 0
    }

    for node in graph.nodes():
        strategy = graph.nodes[node]["strategy"]

        if graph.nodes[node]["alive"]:
            counts[strategy] += 1

    return counts


def average_energy_by_strategy(graph):
    """
    Calculates average energy for each strategy.

    Dead agents remain included with zero energy so that the result
    reflects both survival and resource accumulation.
    """

    averages = {}

    for strategy in STRATEGIES:
        energies = [
            graph.nodes[node]["energy"]
            for node in graph.nodes()
            if graph.nodes[node]["strategy"] == strategy
        ]

        averages[strategy] = np.mean(energies) if energies else 0

    return averages


# -----------------------------
# 5. VISUALIZE THE NETWORK
# -----------------------------

def observe(graph, positions, step):
    """
    Displays the network.

    Node color:
        Green = helper
        Gray = neutral
        Red = aggressive
        Black = dead

    Node size:
        Larger nodes have more energy.
    """

    color_map = {
        "helper": "green",
        "neutral": "gray",
        "aggressive": "red"
    }

    node_colors = []
    node_sizes = []

    for node in graph.nodes():
        alive = graph.nodes[node]["alive"]
        strategy = graph.nodes[node]["strategy"]
        energy = graph.nodes[node]["energy"]

        if alive:
            node_colors.append(color_map[strategy])
            node_sizes.append(max(150, energy * 35))
        else:
            node_colors.append("black")
            node_sizes.append(100)

    plt.figure(figsize=(9, 7))

    nx.draw(
        graph,
        positions,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color="lightgray",
        font_size=8,
        font_color="white"
    )

    alive_count = sum(
        graph.nodes[node]["alive"]
        for node in graph.nodes()
    )

    plt.title(
        f"Agent-Based Network Model at Step {step}\n"
        f"Living Agents: {alive_count} of {N}"
    )

    plt.tight_layout()
    plt.show()


# -----------------------------
# 6. UPDATE RULES
# -----------------------------

def update(graph):
    """
    Performs one simulation step.

    Step order:
    1. Living agents gain environmental energy.
    2. Living agents pay a metabolism cost.
    3. Helpers, neutral agents, and aggressive agents act.
    4. Agents with zero energy die.
    """

    # -----------------------------------------
    # Environmental income and survival cost
    # -----------------------------------------

    for node in graph.nodes():
        if graph.nodes[node]["alive"]:
            graph.nodes[node]["energy"] += ENVIRONMENT_INCOME
            graph.nodes[node]["energy"] -= METABOLISM_COST

    # -----------------------------------------
    # Local interactions
    # -----------------------------------------

    nodes = list(graph.nodes())
    random.shuffle(nodes)

    for node in nodes:

        if not graph.nodes[node]["alive"]:
            continue

        living_neighbors = [
            neighbor
            for neighbor in graph.neighbors(node)
            if graph.nodes[neighbor]["alive"]
        ]

        if len(living_neighbors) == 0:
            continue

        target = random.choice(living_neighbors)
        strategy = graph.nodes[node]["strategy"]

        # -----------------------------
        # Helper behavior
        # -----------------------------

        if strategy == "helper":

            # Helpers give only if they can afford to share.
            if graph.nodes[node]["energy"] > HELP_AMOUNT + 1:
                graph.nodes[node]["energy"] -= HELP_AMOUNT
                graph.nodes[target]["energy"] += HELP_AMOUNT

        # -----------------------------
        # Aggressive behavior
        # -----------------------------

        elif strategy == "aggressive":

            amount_taken = min(
                STEAL_AMOUNT,
                graph.nodes[target]["energy"]
            )

            graph.nodes[target]["energy"] -= amount_taken
            graph.nodes[node]["energy"] += amount_taken

        # -----------------------------
        # Neutral behavior
        # -----------------------------

        elif strategy == "neutral":
            pass

    # -----------------------------------------
    # Death rule
    # -----------------------------------------

    for node in graph.nodes():

        if graph.nodes[node]["energy"] <= 0:
            graph.nodes[node]["energy"] = 0
            graph.nodes[node]["alive"] = False


# -----------------------------
# 7. RUN THE SIMULATION
# -----------------------------

def run_simulation():
    """
    Creates the world, runs the model, records results,
    and displays graphs.
    """

    graph = create_connected_random_graph(N, C)
    initialize_agents(graph)

    # Use one fixed layout for all observations.
    positions = nx.spring_layout(graph, seed=RANDOM_SEED)

    # Overall histories
    average_energy_history = []
    alive_history = []
    inequality_history = []

    # Strategy energy histories
    helper_energy_history = []
    neutral_energy_history = []
    aggressive_energy_history = []

    # Strategy survival histories
    helper_alive_history = []
    neutral_alive_history = []
    aggressive_alive_history = []

    # Display the initial network
    if 0 in OBSERVATION_STEPS:
        observe(graph, positions, step=0)

    for step in range(1, STEPS + 1):

        update(graph)

        energies = [
            graph.nodes[node]["energy"]
            for node in graph.nodes()
        ]

        alive_count = sum(
            graph.nodes[node]["alive"]
            for node in graph.nodes()
        )

        strategy_energies = average_energy_by_strategy(graph)
        strategy_alive = count_alive_by_strategy(graph)

        # Record overall measurements
        average_energy_history.append(np.mean(energies))
        alive_history.append(alive_count)
        inequality_history.append(gini(energies))

        # Record energy by strategy
        helper_energy_history.append(strategy_energies["helper"])
        neutral_energy_history.append(strategy_energies["neutral"])
        aggressive_energy_history.append(strategy_energies["aggressive"])

        # Record survival by strategy
        helper_alive_history.append(strategy_alive["helper"])
        neutral_alive_history.append(strategy_alive["neutral"])
        aggressive_alive_history.append(strategy_alive["aggressive"])

        if step in OBSERVATION_STEPS:
            observe(graph, positions, step)

    # -----------------------------
    # 8. PRINT FINAL RESULTS
    # -----------------------------

    final_strategy_energy = average_energy_by_strategy(graph)
    final_strategy_alive = count_alive_by_strategy(graph)

    print("\nFINAL RESULTS")
    print("-" * 45)

    print(f"Total living agents: {alive_history[-1]} out of {N}")
    print(f"Final average energy: {average_energy_history[-1]:.2f}")
    print(f"Final Gini inequality: {inequality_history[-1]:.3f}")

    print("\nAverage energy by strategy:")

    for strategy in STRATEGIES:
        print(
            f"  {strategy.capitalize():10s}: "
            f"{final_strategy_energy[strategy]:.2f}"
        )

    print("\nLiving agents by strategy:")

    for strategy in STRATEGIES:
        print(
            f"  {strategy.capitalize():10s}: "
            f"{final_strategy_alive[strategy]}"
        )

    # -----------------------------
    # 9. PLOT LARGE-SCALE PATTERNS
    # -----------------------------

    time_steps = range(1, STEPS + 1)

    # Average energy
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, average_energy_history)
    plt.xlabel("Time Step")
    plt.ylabel("Average Energy")
    plt.title("Average Energy Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Survival
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, alive_history)
    plt.xlabel("Time Step")
    plt.ylabel("Number of Living Agents")
    plt.title("Agent Survival Over Time")
    plt.ylim(0, N + 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Inequality
    plt.figure(figsize=(8, 5))
    plt.plot(time_steps, inequality_history)
    plt.xlabel("Time Step")
    plt.ylabel("Gini Index")
    plt.title("Energy Inequality Over Time")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Average energy by strategy
    plt.figure(figsize=(8, 5))
    plt.plot(
        time_steps,
        helper_energy_history,
        label="Helpers",
        color="green"
    )
    plt.plot(
        time_steps,
        neutral_energy_history,
        label="Neutral",
        color="gray"
    )
    plt.plot(
        time_steps,
        aggressive_energy_history,
        label="Aggressive",
        color="red"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Average Energy")
    plt.title("Average Energy by Strategy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Survival by strategy
    plt.figure(figsize=(8, 5))
    plt.plot(
        time_steps,
        helper_alive_history,
        label="Helpers",
        color="green"
    )
    plt.plot(
        time_steps,
        neutral_alive_history,
        label="Neutral",
        color="gray"
    )
    plt.plot(
        time_steps,
        aggressive_alive_history,
        label="Aggressive",
        color="red"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Number of Living Agents")
    plt.title("Survival by Strategy")
    plt.ylim(0, N // 3 + 2)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------------
# 10. START THE PROGRAM
# -----------------------------

if __name__ == "__main__":
    run_simulation()
