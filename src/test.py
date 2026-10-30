import random
import math
import os
import csv

STATE_FILENAME = "bt_sorting_state.csv"

# ==========================================
# 1. INTERACTIVE PROMPT HELPER
# ==========================================
def ask_user(item1, item2):
    print(f"\n[Comparison] Which value is greater/better?")
    print(f"  [1] {item1}")
    print(f"  [2] {item2}")
    print(f"  [q] Save and Exit")
    while True:
        choice = input("Enter 1, 2, or q: ").strip().lower()
        if choice == '1':
            return True
        elif choice == '2':
            return False
        elif choice == 'q':
            return 'EXIT'
        print("Invalid input. Please enter 1, 2, or q.")


# ==========================================
# 2. CONVERGENCE CHECK (Spearman's Rho)
# ==========================================
def calculate_spearman_rho(ranking_old, ranking_new):
    n = len(ranking_new)
    if n <= 1:
        return 1.0
    
    old_ranks = {item: idx for idx, item in enumerate(ranking_old)}
    new_ranks = {item: idx for idx, item in enumerate(ranking_new)}
    
    d_squared_sum = sum((old_ranks[item] - new_ranks[item]) ** 2 for item in ranking_new)
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho


# ==========================================
# NEW: UNCERTAINTY-WEIGHTED PRIORITY PAIRING
# ==========================================
def get_uncertainty_weighted_pair(items, bt_scores, match_counts, window=2):
    """
    Picks an item heavily favoring those with fewer matches (higher uncertainty),
    then pairs it with its closest scoring neighbor.
    """
    n = len(items)
    sorted_items = sorted(items, key=lambda x: bt_scores[x], reverse=True)
    
    # Calculate uncertainty weights (inverse of match count)
    # Fewer matches = higher weight (higher chance of being picked next)
    weights = [1.0 / (match_counts[item] + 1.0) for item in sorted_items]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    
    # Select item1 using the uncertainty distribution
    item1 = random.choices(sorted_items, weights=probabilities, k=1)[0]
    idx1 = sorted_items.index(item1)
    
    # Find closest scoring neighbors around item1
    low_bound = max(0, idx1 - window)
    high_bound = min(n - 1, idx1 + window)
    
    neighbors = [sorted_items[i] for i in range(low_bound, high_bound + 1) if sorted_items[i] != item1]
    if not neighbors:
        neighbors = [item for item in sorted_items if item != item1]
        
    item2 = random.choice(neighbors)
    return item1, item2


# ==========================================
# 3. STATE PERSISTENCE (Dynamic Add & Drop)
# ==========================================
def save_state(filename, items, wins, matches):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["ITEMS"] + list(map(str, items)))
        
        writer.writerow(["WINS_MATRIX"])
        for row in wins:
            writer.writerow(row)
            
        writer.writerow(["MATCHES_MATRIX"])
        for row in matches:
            writer.writerow(row)
    print(f"\n[State Saved] Progress successfully saved to '{filename}'.")


def load_state(filename, default_items):
    """
    Loads state from CSV. Automatically handles:
    1. Pruning items that were removed from default_items.
    2. Expanding matrices for newly added items with default priors.
    3. Preserving historical match data for items that remain.
    """
    if not os.path.exists(filename):
        print(f"\n[Notice] No saved state found at '{filename}'. Starting fresh.")
        n = len(default_items)
        wins = [[0.1]*n for _ in range(n)]
        matches = [[0.2]*n for _ in range(n)]
        for i in range(n):
            wins[i][i] = 0.0
            matches[i][i] = 0.0
        return default_items, wins, matches
    
    print(f"\n[Loading State] Found saved state at '{filename}'...")
    with open(filename, mode='r', newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        
    saved_items = reader[0][1:]
    
    wins = []
    matches = []
    mode = None
    for row in reader:
        if not row:
            continue
        if row[0] == "WINS_MATRIX":
            mode = "wins"
            continue
        elif row[0] == "MATCHES_MATRIX":
            mode = "matches"
            continue
            
        if mode == "wins":
            wins.append([float(x) for x in row])
        elif mode == "matches":
            matches.append([float(x) for x in row])
            
    saved_item_to_idx = {item: i for i, item in enumerate(saved_items)}
    
    # Check for added and removed items
    removed_items = [item for item in saved_items if item not in default_items]
    added_items = [item for item in default_items if item not in saved_items]
    
    if removed_items:
        print(f"\n[Notice] Detected dropped items: {removed_items}. Pruning from state...")
    if added_items:
        print(f"\n[Notice] Detected new items added: {added_items}. Expanding state...")
        
    # Build new matrices mapped strictly to default_items order
    n = len(default_items)
    new_wins = [[0.1]*n for _ in range(n)]
    new_matches = [[0.2]*n for _ in range(n)]
    for i in range(n):
        new_wins[i][i] = 0.0
        new_matches[i][i] = 0.0
        
    # Safely copy historical data over for items existing in both lists
    for i, item_i in enumerate(default_items):
        if item_i in saved_item_to_idx:
            old_i = saved_item_to_idx[item_i]
            for j, item_j in enumerate(default_items):
                if item_j in saved_item_to_idx:
                    old_j = saved_item_to_idx[item_j]
                    new_wins[i][j] = wins[old_i][old_j]
                    new_matches[i][j] = matches[old_i][old_j]
                    
    return default_items, new_wins, new_matches


# ==========================================
# 4. BRADLEY-TERRY MODEL (With Uncertainty Pairing & Tolerance)
# ==========================================
def bradley_terry_sort_interactive(items, check_interval=5, stability_threshold=0.88):
    """
    Sorts items using Bradley-Terry with uncertainty-weighted pairing and configurable tolerance.
    """
    print("\n--- Starting Bradley-Terry Model Sort (Uncertainty-Weighted + Tolerant) ---")
    
    items, wins, matches = load_state(STATE_FILENAME, items)
    n_items = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}
    
    match_counts = {item: sum(matches[item_to_idx[item]]) for item in items}
    p = [1.0] * n_items
    
    total_matches_logged = sum(sum(row) for row in matches)
    if total_matches_logged > 0:
        for _ in range(40):
            new_p = [0.0] * n_items
            for i in range(n_items):
                total_wins_i = sum(wins[i])
                denominator = 0.0
                for j in range(n_items):
                    if i != j and matches[i][j] > 0:
                        denominator += matches[i][j] / max(p[i] + p[j], 1e-9)
                if denominator > 0:
                    new_p[i] = total_wins_i / denominator
                else:
                    new_p[i] = p[i]
            norm = sum(new_p)
            if norm > 0:
                p = [max(val / norm, 1e-6) for val in new_p]

    comparisons = int(total_matches_logged / 2)
    previous_ranking = [items[i] for i in sorted(range(n_items), key=lambda x: p[x], reverse=True)]
    stable_streak = 0
    
    while True:
        bt_scores = {items[i]: p[i] for i in range(n_items)}
        
        # USE UNCERTAINTY-WEIGHTED PAIRING HERE
        item1, item2 = get_uncertainty_weighted_pair(items, bt_scores, match_counts, window=2)
        
        idx1, idx2 = item_to_idx[item1], item_to_idx[item2]
        
        result = ask_user(item1, item2)
        if result == 'EXIT':
            save_state(STATE_FILENAME, items, wins, matches)
            print("\n[Exited Gracefully] You can resume anytime by rerunning the app.")
            return None, comparisons
            
        item1_wins = result
        comparisons += 1
        
        matches[idx1][idx2] += 1
        matches[idx2][idx1] += 1
        match_counts[item1] += 1
        match_counts[item2] += 1
        
        if item1_wins:
            wins[idx1][idx2] += 1
        else:
            wins[idx2][idx1] += 1
            
        for _ in range(30):
            new_p = [0.0] * n_items
            for i in range(n_items):
                total_wins_i = sum(wins[i])
                denominator = 0.0
                for j in range(n_items):
                    if i != j and matches[i][j] > 0:
                        denominator += matches[i][j] / max(p[i] + p[j], 1e-9)
                if denominator > 0:
                    new_p[i] = total_wins_i / denominator
                else:
                    new_p[i] = p[i]
            
            norm = sum(new_p)
            if norm > 0:
                p = [max(val / norm, 1e-6) for val in new_p]
                
        if comparisons % check_interval == 0:
            current_ranking = [items[i] for i in sorted(range(n_items), key=lambda x: p[x], reverse=True)]
            rho = calculate_spearman_rho(previous_ranking, current_ranking)
            print(f"\n[Convergence Check] Comparisons: {comparisons} | Rank Stability (Rho): {rho:.3f}")
            
            min_matches = min(match_counts.values())
            
            # STRICTER CONVERGENCE GATES:
            # 1. Every item must have at least 10 matches so no item is half-baked.
            # 2. Must maintain high stability across 3 consecutive checks.
            if min_matches >= 10 and rho >= stability_threshold:
                stable_streak += 1
                if stable_streak >= 3:
                    print("\n[Convergence Achieved!] Rankings have thoroughly stabilized.")
                    break
            else:
                stable_streak = 0
                
            previous_ranking = current_ranking
            
        if comparisons >= len(items) * 20:
            print("\n[Notice] Reached maximum allocated safety comparisons.")
            break
            
    final_ranking = [items[i] for i in sorted(range(n_items), key=lambda x: p[x], reverse=True)]
    
    print(f"\n==========================================")
    print(f" FINAL RANKINGS & LOG-ABILITY SCORES")
    print(f"==========================================")
    for rank, item in enumerate(final_ranking, start=1):
        idx = item_to_idx[item]
        log_score = math.log(max(p[idx], 1e-6))
        matches_played = int(match_counts[item])
        print(f"Rank {rank:2d}: {item} (Log-Ability: {log_score:.2f}, Matches: {matches_played})")
    print(f"==========================================")
    
    save_state(STATE_FILENAME, items, wins, matches)
    return final_ranking, comparisons


# ==========================================
# EXAMPLE EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    sample_items = [f"image_{i:02d}.png" for i in range(1, 11)]
    sorted_ranking, total_comparisons = bradley_terry_sort_interactive(sample_items)
    
    if sorted_ranking:
        print(f"\n[Result] Sorting completed with {total_comparisons} total comparisons.")