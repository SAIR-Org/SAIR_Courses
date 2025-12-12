#!/bin/bash
# Simple CLI installation - create direct symlinks

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$HOME/.local/bin"

echo "Installing CLI tools to $BIN_DIR"
mkdir -p "$BIN_DIR"

# Create symlinks to the actual Python scripts
ln -sf "$PROJECT_DIR/scripts/run_pipeline.py" "$BIN_DIR/dl-pipeline"
ln -sf "$PROJECT_DIR/scripts/cli_tools.py" "$BIN_DIR/dl-tools"

# Create wrapper scripts that call the appropriate functions
cat > "$BIN_DIR/dl-download" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/cli_tools.py download "$@"
SCRIPT

cat > "$BIN_DIR/dl-train" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/cli_tools.py train "$@"
SCRIPT

cat > "$BIN_DIR/dl-ui" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/cli_tools.py ui "$@"
SCRIPT

cat > "$BIN_DIR/dl-evaluate" << 'SCRIPT'
#!/bin/bash
cd "$PROJECT_DIR"
python scripts/cli_tools.py evaluate "$@"
SCRIPT

# Make them executable
chmod +x "$BIN_DIR"/dl-*

echo "✅ CLI tools installed!"
echo ""
echo "Available commands:"
echo "  dl-pipeline  -- Run complete pipeline (dl-pipeline --help)"
echo "  dl-download  -- Download datasets (dl-download --help)"
echo "  dl-train     -- Train a specific model (dl-train --help)"
echo "  dl-ui        -- Launch inference UI (dl-ui --help)"
echo "  dl-evaluate  -- Evaluate existing models (dl-evaluate --help)"
echo ""
echo "Usage examples:"
echo "  dl-download --datasets mnist fashion cifar10"
echo "  dl-train --dataset mnist --architecture deep_dnn"
echo "  dl-pipeline --run-mode evaluate"
echo "  dl-ui --share-ui"
