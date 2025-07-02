# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building and Installation
- `python setup.py develop` - Install PyTorch/XLA in development mode (editable install)
- `scripts/build_developer.sh` - Rebuild torch_xla (including torchax)
- `scripts/build_developer.sh -b pytorch` - Rebuild PyTorch, TorchVision, and PyTorch/XLA
- `scripts/build_developer.sh -b vision` - Rebuild TorchVision and PyTorch/XLA
- `scripts/build_developer.sh -t` - Test that libraries are installed correctly
- `bazel build //...` - Build all targets using Bazel
- `bazel clean --expunge` - Clean all build artifacts

### Testing
- `test/run_tests.sh` - Run comprehensive Python test suite
- `test/cpp/run_tests.sh` - Run C++ tests
- `PJRT_DEVICE=CPU test/run_tests.sh` - Run tests on CPU
- `PJRT_DEVICE=TPU test/run_tests.sh` - Run tests on TPU
- `PJRT_DEVICE=CUDA GPU_NUM_DEVICES=4 test/run_tests.sh` - Run tests on GPU
- Individual test shards can be run with environment variables like `RUN_XLA_OP_TESTS1=xla_op1`

### Code Quality
- `scripts/git_fix.py` - Format C++ and Python files according to project standards
- `scripts/git_fix.py --set_git_push_hook` - Set up automatic formatting on git push
- Uses `clang-format-11` for C++ and `yapf==0.40.2` for Python formatting

### Development Tools
- `scripts/update_compile_commands.py` - Generate compilation database for clangd/IDE support
- `scripts/update_deps.py` - Update libtpu, JAX, and other dependencies
- `scripts/apply_patches.sh` - Apply OpenXLA patches

## Code Architecture

### High-Level Structure
PyTorch/XLA bridges PyTorch and XLA compiler, enabling PyTorch models to run on TPUs, GPUs, and other XLA-supported accelerators.

**Core Components:**
- `torch_xla/csrc/` - C++ implementation containing the core XLA integration
- `torch_xla/core/` - Python core functionality and XLA model interface  
- `torch_xla/distributed/` - Distributed training support (SPMD, FSDP, DDP)
- `torch_xla/experimental/` - Experimental features (Pallas kernels, eager mode, etc.)
- `torchax/` - JAX interoperability layer

### C++ Backend (`torch_xla/csrc/`)
- **Runtime layer** (`runtime/`) - Manages computation clients (PJRT, IFRT), device coordination
- **IR layer** (`ir.cpp`, `lowering_context.cpp`) - Intermediate representation and lowering to XLA HLO
- **Operations** (`ops/`) - XLA implementations of PyTorch operations
- **Tensor implementation** (`tensor.cpp`, `tensor_impl.cpp`) - XLA tensor backend
- **Device management** (`device.cpp`) - XLA device abstraction

### Python Frontend (`torch_xla/`)
- **Core API** (`core/xla_model.py`) - Main user-facing functions like `xm.mark_step()`, device management
- **Distributed training** (`distributed/`) - SPMD sharding, FSDP, parallel data loading
- **Dynamo integration** (`_dynamo/`) - torch.compile support for XLA compilation
- **Debugging tools** (`debug/`) - Profiling, metrics, graph visualization

### Build System
- **Bazel** - Primary build system, configured in `BUILD` files throughout the codebase
- **setup.py** - Python packaging, integrates with Bazel for C++ extensions
- **Dependencies** - JAX for some operations, libtpu for TPU support, OpenXLA for core compiler

### Key Integration Points
- **PyTorch ATen integration** - Custom dispatcher and tensor implementations
- **XLA/OpenXLA** - Uses XLA compiler backend for optimized execution
- **PJRT** - Runtime interface for distributed execution across accelerators
- **Plugin system** - Supports TPU, CUDA, CPU plugins via entry points

### Development Workflow Patterns
1. **New operations** - Add to `torch_xla/csrc/ops/`, register in `ops.cpp`, test in `test/test_operations.py`
2. **Bug fixes** - Modify C++ implementation, ensure Python tests pass
3. **Performance features** - Often implemented in experimental/ first, then moved to core
4. **Device support** - Add new plugins in `plugins/` directory with appropriate build configs

## Environment Variables

### Runtime Configuration
- `PJRT_DEVICE` - Select device type (CPU, TPU, CUDA)
- `GPU_NUM_DEVICES` - Number of GPU devices to use
- `CPU_NUM_DEVICES` - Number of CPU devices to use
- `XLA_USE_SPMD=1` - Enable SPMD mode for distributed training
- `XLA_USE_BF16=1` - Use bfloat16 precision
- `XLA_EXPERIMENTAL` - Enable experimental features

### Debugging
- `PT_XLA_DEBUG=1` - Enable debug logging
- `XLA_DUMP_FATAL_STACK=1` - Dump stack traces on fatal errors
- `XLA_USE_EAGER_DEBUG_MODE=1` - Enable eager debug mode

### Build Configuration
- `DEBUG=1` - Build with debug symbols
- `BUILD_CPP_TESTS=1` - Build C++ tests
- `XLA_CUDA=1` - Build with CUDA support
- `BUNDLE_LIBTPU=1` - Include libtpu in wheel

## Repository Structure Notes
- Master branch is `master` (not `main`)
- Uses git submodules for some dependencies
- Docker development environment available in `.devcontainer/`
- Comprehensive documentation in `docs/source/` with Sphinx build system
- Examples and reference implementations in `examples/` directory