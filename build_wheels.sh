#!/bin/bash

set -e

# Build wheels for Python packages
# Modify the PACKAGES array below to change which packages to build
# Format: "package" or "package=version"

# Package list (edit this array)
PACKAGES=(
    "bashlex==0.18"
)

PYTHON_VERSION="3.12"
VERBOSE_LEVEL=1
BUILD_WHEEL_SCRIPT="./wheels_builder/build_wheel.sh"

cd "$(dirname "$0")/deps"

# Check if build script exists
if [[ ! -f "$BUILD_WHEEL_SCRIPT" ]]; then
    echo "ERROR: $BUILD_WHEEL_SCRIPT not found"
    cd "$(dirname "$0")"
    git submodule init
    git submodule sync
    git submodule update

    if [[ ! -f "$BUILD_WHEEL_SCRIPT" ]]; then
        echo "ERROR: $BUILD_WHEEL_SCRIPT not found"
        exit 1
    fi
    echo "Solved by setting up submodules"
fi

# Check if package list is empty
if [[ ${#PACKAGES[@]} -eq 0 ]]; then
    echo "ERROR: No packages defined in PACKAGES array"
    exit 1
fi

echo "==============================================="
echo "Building wheels for Python $PYTHON_VERSION"
echo "==============================================="

# Counter for success/failure
TOTAL=0
SUCCESS=0
FAILED=0
FAILED_PACKAGES=""

# Process each package in the PACKAGES array
for package_spec in "${PACKAGES[@]}"; do
    TOTAL=$((TOTAL + 1))

    # Parse package name and version
    if [[ "$package_spec" =~ ^([^=]+)=(.+)$ ]]; then
        PACKAGE="${BASH_REMATCH[1]}"
        VERSION="${BASH_REMATCH[2]}"
        echo ""
        echo ">>> Building $PACKAGE==$VERSION"
    else
        PACKAGE="$package_spec"
        VERSION=""
        echo ""
        echo ">>> Building $PACKAGE (latest)"
    fi

    # Build the command
    CMD="bash $BUILD_WHEEL_SCRIPT --package $PACKAGE --python=$PYTHON_VERSION --verbose=$VERBOSE_LEVEL"

    # Add version if specified
    if [[ -n "$VERSION" ]]; then
        CMD="$CMD --version $VERSION"
    fi

    # Execute build
    if eval "$CMD"; then
        SUCCESS=$((SUCCESS + 1))
        echo "✓ Successfully built $PACKAGE"
    else
        FAILED=$((FAILED + 1))
        FAILED_PACKAGES="$FAILED_PACKAGES $PACKAGE"
        echo "✗ Failed to build $PACKAGE"
    fi
done

# Summary
echo ""
echo "==============================================="
echo "Build Summary"
echo "==============================================="
echo "Total: $TOTAL"
echo "Success: $SUCCESS"
echo "Failed: $FAILED"

if [[ $FAILED -gt 0 ]]; then
    echo "Failed packages:$FAILED_PACKAGES"
    exit 1
fi

echo ""
echo "All wheels built and saved to $(pwd)"
