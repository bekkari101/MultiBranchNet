"""Debug script to trace parameter calculations"""
from hyperP import DEFAULT_MODEL_CONFIG
from model import ParallelBranchNet

config = DEFAULT_MODEL_CONFIG

print("=" * 80)
print("DETAILED PARAMETER BREAKDOWN")
print("=" * 80)

# Manual calculation
print("\n1. CONVOLUTIONAL LAYERS:")
print(f"   Input channels: {config.input_channels}")
conv_params = 0
prev_ch = config.input_channels
for i in range(config.num_conv_layers):
    curr_ch = config.conv_channels[i]
    weights = prev_ch * curr_ch * config.conv_kernel_size * config.conv_kernel_size
    bias = curr_ch
    layer_params = weights + bias
    conv_params += layer_params
    print(f"   Layer {i+1}: {prev_ch}→{curr_ch}, kernel {config.conv_kernel_size}x{config.conv_kernel_size}")
    print(f"      Weights: {prev_ch} × {curr_ch} × {config.conv_kernel_size} × {config.conv_kernel_size} = {weights:,}")
    print(f"      Bias: {bias}")
    print(f"      Total: {layer_params:,}")
    prev_ch = curr_ch

print(f"\n   TOTAL CONV PARAMS: {conv_params:,}")

# Branch layers
print("\n2. BRANCH LAYERS:")
print(f"   Flat size after conv: {config.flat_size}")
print(f"   Padding needed: {config.pad_size}")
print(f"   Padded size: {config.flat_size + config.pad_size}")

branch_params = 0
for layer_idx, layer_config in enumerate(config.branch_layers):
    print(f"\n   Layer {layer_idx + 1}:")
    print(f"      Num branches: {layer_config.num_branches}")
    print(f"      Nodes per branch: {layer_config.nodes_per_branch}")
    
    if layer_idx == 0:
        padded_size = config.flat_size + config.pad_size
        input_per_branch = padded_size // layer_config.num_branches
        print(f"      Input per branch: {padded_size} / {layer_config.num_branches} = {input_per_branch}")
    else:
        prev_layer_config = config.branch_layers[layer_idx - 1]
        if prev_layer_config.num_branches % layer_config.num_branches == 0:
            branches_per_group = prev_layer_config.num_branches // layer_config.num_branches
            input_per_branch = branches_per_group * prev_layer_config.nodes_per_branch
            print(f"      Structured connectivity: {branches_per_group} branches → 1")
            print(f"      Input per branch: {branches_per_group} × {prev_layer_config.nodes_per_branch} = {input_per_branch}")
        else:
            input_per_branch = prev_layer_config.total_nodes // layer_config.num_branches
            print(f"      Full connectivity: {input_per_branch} inputs per branch")
    
    # Each branch: Linear(input_per_branch, nodes_per_branch)
    branch_weights = input_per_branch * layer_config.nodes_per_branch
    branch_bias = layer_config.nodes_per_branch
    single_branch_params = branch_weights + branch_bias
    layer_params = layer_config.num_branches * single_branch_params
    
    print(f"      Per branch: {input_per_branch} × {layer_config.nodes_per_branch} + {layer_config.nodes_per_branch} = {single_branch_params:,}")
    print(f"      All branches: {layer_config.num_branches} × {single_branch_params:,} = {layer_params:,}")
    
    branch_params += layer_params

print(f"\n   TOTAL BRANCH PARAMS: {branch_params:,}")

# Output layer
print("\n3. OUTPUT LAYER:")
output_weights = config.final_hidden_size * config.num_classes
output_bias = config.num_classes
output_params = output_weights + output_bias
print(f"   Input size: {config.final_hidden_size}")
print(f"   Output size: {config.num_classes}")
print(f"   Weights: {config.final_hidden_size} × {config.num_classes} = {output_weights:,}")
print(f"   Bias: {output_bias}")
print(f"   Total: {output_params:,}")

# Total
estimated_total = conv_params + branch_params + output_params
print("\n" + "=" * 80)
print(f"ESTIMATED TOTAL: {estimated_total:,}")
print(f"   Conv:   {conv_params:,}")
print(f"   Branch: {branch_params:,}")
print(f"   Output: {output_params:,}")
print("=" * 80)

# Now count actual
print("\nBuilding actual model and counting parameters...")
model = ParallelBranchNet(config)

print("\nACTUAL MODEL PARAMETERS:")
actual_total = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"   {name}: {param.numel():,} ({list(param.shape)})")
        actual_total += param.numel()

print("\n" + "=" * 80)
print(f"ACTUAL TOTAL: {actual_total:,}")
print("=" * 80)

print(f"\nDIFFERENCE: {actual_total - estimated_total:+,}")
if actual_total != estimated_total:
    print(f"ERROR: {abs(actual_total - estimated_total) / actual_total * 100:.2f}% mismatch!")
else:
    print("✅ PERFECT MATCH!")
