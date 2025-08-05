import torch
import matplotlib.pyplot as plt

# Modified entropy calculation function with fixed random seed
def calc_entropy(flat_tensor: torch.Tensor, sample_size: int = 100000000, seed: int = 42) -> float:
    global Pioneer
    if Pioneer:
        print("calculating entropy")
    
    with torch.no_grad():
        torch.manual_seed(seed)
        if flat_tensor.numel() <= sample_size:
            sample = flat_tensor
        else:
            indices = torch.randperm(flat_tensor.numel(), generator=torch.Generator().manual_seed(seed))[:sample_size]
            sample = flat_tensor[indices]
        
        unique_vals, counts = sample.unique(return_counts=True)
        probs = counts.float() / counts.sum()
        entropy = -torch.sum(probs * torch.log2(probs)).item()
    
    return entropy

# Set Pioneer flag
Pioneer = True

# Generate synthetic large dataset
def generate_synthetic_data(size: int, num_categories: int = 100) -> torch.Tensor:
    # Simulate a large dataset with a categorical distribution
    probs = torch.rand(num_categories)
    probs = probs / probs.sum()  # Normalize to probabilities
    return torch.multinomial(probs, size, replacement=True)

# Verification function
def verify_entropy_convergence(data_size: int = 10000000, sample_sizes: list = None, seed: int = 42, num_trials: int = 5):
    if sample_sizes is None:
        sample_sizes = [1000, 10000, 50000, 100000, 500000, 1000000]
    
    # Generate large dataset
    print(f"Generating dataset of size {data_size}...")
    data = generate_synthetic_data(data_size).float()
    
    # Calculate true entropy (full dataset)
    true_entropy = calc_entropy(data, sample_size=data_size, seed=seed)
    print(f"True entropy (full dataset): {true_entropy:.6f}")
    
    # Calculate entropy for different sample sizes
    entropy_results = {size: [] for size in sample_sizes}
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        for sample_size in sample_sizes:
            entropy = calc_entropy(data, sample_size=sample_size, seed=seed + trial)
            entropy_results[sample_size].append(entropy)
            print(f"Sample size {sample_size}: Entropy = {entropy:.6f}")
    
    # Calculate mean and standard deviation for each sample size
    mean_entropies = [sum(entropy_results[size]) / num_trials for size in sample_sizes]
    std_entropies = [
        (sum((x - sum(entropy_results[size]) / num_trials) ** 2 for x in entropy_results[size]) / num_trials) ** 0.5
        for size in sample_sizes
    ]
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(sample_sizes, mean_entropies, yerr=std_entropies, fmt='o-', capsize=5, label='Sampled Entropy')
    plt.axhline(y=true_entropy, color='r', linestyle='--', label='True Entropy')
    plt.xscale('log')
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Entropy (bits)')
    plt.title('Entropy vs Sample Size (with Standard Deviation)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print summary
    print("\nSummary:")
    for size, mean, std in zip(sample_sizes, mean_entropies, std_entropies):
        error = abs(mean - true_entropy)
        print(f"Sample size {size}: Mean Entropy = {mean:.6f}, Std = {std:.6f}, Error = {error:.6f}")

# Run verification
if __name__ == "__main__":
    # Set parameters
    data_size = 10000000  # 10M elements
    sample_sizes = [1000, 10000, 50000, 100000, 500000, 1000000]
    num_trials = 5
    
    # Run verification
    verify_entropy_convergence(data_size=data_size, sample_sizes=sample_sizes, seed=42, num_trials=num_trials)