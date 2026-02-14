# Modules Directory

This directory contains external verifier submodules integrated into the the framework framework. These are established neural network verification tools that the framework provides unified interfaces for.

## Submodules Overview

### αβ-CROWN (`abcrown/`)
**Complete Neural Network Verifier with Branch-and-Bound**

- **Repository**: https://github.com/Verified-Intelligence/alpha-beta-CROWN
- **License**: BSD 3-Clause License
- **Integration**: Git submodule
- **Purpose**: State-of-the-art complete verification with advanced branch-and-bound algorithms

#### the framework Integration Notes:
- Parameters mapped to the framework unified interface
- Configuration managed through the framework's parameter system
- Compatible with the framework's specification refinement framework

### ERAN (`eran/`)
**ETH Robustness Analyzer for Neural Networks**

- **Repository**: https://github.com/eth-sri/eran
- **License**: Apache 2.0 License
- **Integration**: Git submodule
- **Purpose**: Abstract interpretation-based verification with multiple domains

#### Abstract Domains Supported:
- **DeepPoly**: Polyhedra-based abstract interpretation
- **DeepZono**: Zonotope-based abstract interpretation
- **RefinePoly**: Refinement-based DeepPoly with MILP
- **RefineZono**: Refinement-based DeepZono with optimization

#### the framework Integration Notes:
- Environment isolation due to Python 3.8/TensorFlow 2.9.3 requirements
- Parameter translation for ERAN's command-line interface

## Submodule Management

### Initialisation
Submodules are automatically initialised during setup:
```bash
git clone --recursive <ANONYMOUS_REPO_URL>
```

### Manual Submodule Updates
```bash
git submodule update --init --recursive
git submodule update --remote
```

### Submodule Status
Check submodule status:
```bash
git submodule status
```

## Integration Architecture

### Parameter Mapping
the framework translates its unified parameters to each tool's native format:
- Common parameters (model, dataset, epsilon) mapped directly
- Tool-specific parameters preserved when using respective backends
- Default values provided for missing parameters

### Environment Isolation
- **ERAN**: Isolated Python 3.8 environment (`cuc-eran`)
- **αβ-CROWN**: Shared Python 3.9 environment (`cuc-abcrown`)
- **the framework**: Main Python 3.9 environment (`cuc-main`)

## Compatibility Matrix

| Feature | ERAN | αβ-CROWN | the framework Native |
|---------|------|----------|------------|
| MNIST | ✓ | ✓ | ✓ |
| CIFAR-10 | ✓ | ✓ | ✓ |
| VNNLIB | x | ✓ | ✓ |
| BaB Refinement | x | ✓ | ✓ |

## Troubleshooting

### Submodule Issues
```bash
# Reset submodules to clean state
git submodule deinit --all -f
git submodule update --init --recursive
```

### ERAN Environment Issues
```bash
# Rebuild ERAN environment
conda env remove -n cuc-eran
cd setup/
source eran_env_setup.sh
```

### αβ-CROWN Import Conflicts
The setup script automatically patches αβ-CROWN imports to prevent conflicts. If issues persist:
```bash
cd modules/abcrown/
git checkout .  # Reset any local changes
cd ../../setup/
source setup.sh   # Re-run setup to reapply patches
```

## Contributing to Submodules

### ERAN Issues
Report issues directly to: https://github.com/eth-sri/eran

### αβ-CROWN Issues  
Report issues directly to: https://github.com/Verified-Intelligence/alpha-beta-CROWN

### Integration Issues
Report framework-specific integration issues to the main repository.
