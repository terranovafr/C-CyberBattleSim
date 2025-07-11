#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025 Franco Terranova.
# Licensed under the MIT License.


set -ex

pushd "$(dirname "$0")"

conda info --envs

eval "$(conda shell.bash hook)"

if conda info --envs | grep -q ccyberbattlesim; then
  echo "env already exists";
else
  conda env create -f environment.yml;
fi

conda activate ccyberbattlesim

python --version

if [ ""$GITHUB_ACTION"" == "" ] && [ -d ".git" ]; then
  echo 'running under a git enlistment -> configure pre-commit checks on every `git push` to run pyright and co'
  if command -v pre-commit &> /dev/null; then
    pre-commit install -t pre-push
  else
    echo "pre-commit not found, skipping hook installation"
  fi
fi

chmod +x setup_files.sh
./setup_files.sh

popd
