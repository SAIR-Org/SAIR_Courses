#!/bin/bash
# Install CLI tools as symbolic links

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"

echo "Installing CLI tools to $BIN_DIR"

# Create bin directory if it doesn't exist
mkdir -p "$BIN_DIR"

# Create wrapper scripts
cat > "$BIN_DIR/dl-pipeline" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/run_pipeline.py "$@"
SCRIPT

cat > "$BIN_DIR/dl-download" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/cli_tools.py download-data "$@"
SCRIPT

cat > "$BIN_DIR/dl-train" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/cli_tools.py train-model "$@"
SCRIPT

cat > "$BIN_DIR/dl-ui" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/cli_tools.py launch-ui "$@"
SCRIPT

cat > "$BIN_DIR/dl-evaluate" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/cli_tools.py evaluate-models "$@"
SCRIPT

# Make them executable
chmod +x "$BIN_DIR"/dl-*

echo "✅ CLI tools installed!"
echo ""
echo "Available commands:"
echo "  dl-pipeline  -- Run complete pipeline"
echo "  dl-download  -- Download datasets"
echo "  dl-train     -- Train a specific model"
echo "  dl-ui        -- Launch inference UI"
echo "  dl-evaluate  -- Evaluate existing models"
echo ""
echo "Make sure $BIN_DIR is in your PATH"
echo "Add to your ~/.bashrc or ~/.zshrc:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
