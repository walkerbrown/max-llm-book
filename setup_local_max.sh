#!/bin/bash
# Source this file to configure the environment for using the locally-built MAX framework.
# Usage: source setup_local_max.sh

VENV="/Users/dylan/Developer/modular/modular/.max+python+max+entrypoints+pipelines.venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: Bazel venv not found at $VENV"
    echo "Build it first with: cd /Users/dylan/Developer/modular/modular && ./bazelw run //max/python/max/entrypoints:pipelines.venv"
    return 1
fi

# Activate the venv (sets VIRTUAL_ENV, updates PATH, etc.)
source "$VENV/bin/activate"

# Set all Mojo/MAX environment variables needed for compilation
export MODULAR_MOJO_MAX_IMPORT_PATH="$VENV/lib/mojo"
export MODULAR_MOJO_MAX_PACKAGE_ROOT="$VENV"
export MODULAR_MAX_PACKAGE_ROOT="$VENV"
export MODULAR_MOJO_MAX_DRIVER_PATH="$VENV/bin/mojo"
export MODULAR_MOJO_MAX_COMPILERRT_PATH="$VENV/lib/libKGENCompilerRTShared.dylib"
export MODULAR_MOJO_MAX_LLD_PATH="$VENV/bin/lld"

echo "Local MAX environment configured:"
echo "  VIRTUAL_ENV=$VIRTUAL_ENV"
echo "  MODULAR_MOJO_MAX_IMPORT_PATH=$MODULAR_MOJO_MAX_IMPORT_PATH"
echo "  MODULAR_MOJO_MAX_DRIVER_PATH=$MODULAR_MOJO_MAX_DRIVER_PATH"
echo "  MODULAR_MOJO_MAX_COMPILERRT_PATH=$MODULAR_MOJO_MAX_COMPILERRT_PATH"
echo "  Python: $(which python3)"
echo ""
echo "Now run: python3 main.py"
