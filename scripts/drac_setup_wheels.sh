#!/bin/bash

RORQUAL_DIR="/home/adefu020/links/projects/def-pbranco/adefu020"
PROJECTS_DIR="/home/adefu020/projects/def-pbranco/adefu020"
if [ -d $RORQUAL_DIR ]; then
    PROJECTS_DIR=$RORQUAL_DIR
fi

ROOT_DIR="$PROJECTS_DIR/TimeWillTell/"
HOME=$PWD
export HOME
export PROJECTS_DIR
export ROOT_DIR

# Setup the tree sitter wheels if they don't exist:
TREE_SITTER_WHEEL_DIR="$PROJECTS_DIR/tree_sitter/"
if [-d $TREE_SITTER_WHEEL_DIR ]; then
    echo "Tree sitter wheels already set up."
else
    cd $PROJECTS_DIR
    git clone "https://github.com/ComputeCanada/wheels_builder.git"
    mkdir -p $TREE_SITTER_WHEEL_DIR
    cd $TREE_SITTER_WHEEL_DIR
    bash $PROJECTS_DIR/wheels_builder/build_wheel.sh --package tree-sitter --python 3.12
    bash $PROJECTS_DIR/wheels_builder/build_wheel.sh --package tree-sitter-c --python 3.12
    bash $PROJECTS_DIR/wheels_builder/build_wheel.sh --package tree-sitter-cpp --python 3.12
fi


