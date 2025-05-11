
# Environment Setup Guide

## 1. Install Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 2. Install Python 3.13

```bash
brew install python@3.13
```

## 3. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3.13 -
```

## 4. Set up Poetry Environment

```bash
# Create symlink for Python
sudo ln -sf /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python

# Initialize Poetry environment
poetry env use python

# Install dependencies
poetry install --no-root

# Activate environment
source $(poetry env info --path)/bin/activate
```
