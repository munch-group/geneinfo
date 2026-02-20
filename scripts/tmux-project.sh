#!/bin/bash

# tmux-project: Advanced wrapper for directory-based tmux sessions with auto-detach
# Usage: tmux-project [directory]

set -e

# Configuration
TARGET_DIR="${1:-$PWD}"
LOCK_DIR="/tmp/tmux-project"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create lock directory
mkdir -p "$LOCK_DIR"

# Change to target directory
if [ -n "$1" ] && [ -d "$1" ]; then
    cd "$1"
elif [ -n "$1" ]; then
    echo -e "${RED}Error:${NC} Directory '$1' does not exist"
    exit 1
fi

# Generate session name based on directory
SESSION_NAME="vscode-$(basename "$PWD")"
LOCK_FILE="$LOCK_DIR/${SESSION_NAME}.lock"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[tmux-project]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[tmux-project]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[tmux-project]${NC} $1"
}

print_error() {
    echo -e "${RED}[tmux-project]${NC} $1"
}

# Function to cleanup on exit
cleanup() {
    print_status "Terminal closing, handling session cleanup..."
    
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        # Check if there are other clients attached
        client_count=$(tmux list-clients -t "$SESSION_NAME" 2>/dev/null | wc -l)
        
        if [ "$client_count" -le 1 ]; then
            print_warning "Last client, detaching session: $SESSION_NAME"
            tmux detach-session -t "$SESSION_NAME" 2>/dev/null || true
        else
            print_status "Other clients attached, just disconnecting this one"
        fi
    fi
    
    rm -f "$LOCK_FILE"
}

# Function to show usage
show_usage() {
    echo "Usage: tmux-project [directory]"
    echo ""
    echo "Creates or attaches to a tmux session named 'vscode-<dirname>'"
    echo ''
    echo 'To use it as the default vscode terminal, you must add this to '
    echo 'your vscode settings.json (you can replace linux with osx):'
    echo ''
    echo ' "terminal.integrated.defaultProfile.linux": "tmux-project",'
    echo ' "terminal.integrated.profiles.linux": {'
    echo '     "tmux-project": {'
    echo '         "path": "/bin/bash",'
    echo '         "args": ["-c", "tmux-project"]'
    echo '     }'
    echo ' },'
    echo ''
    echo 'You can replace'
    echo ''
    echo '         "args": ["-c", "tmux-project"]'
    echo 'with'
    echo '         "args": ["-c", "$PWD/.scripts/tmux-project"]'
    echo 'if you want to add the script to your repository.

    echo "Options:"
    echo "  directory    Target directory (default: current directory)"
    echo ""
    echo "Examples:"
    echo "  tmux-project              # Session: vscode-<current-dir>"
    echo "  tmux-project ~/projects   # Session: vscode-projects"
    echo "  tmux-project /var/log     # Session: vscode-log"
    echo ""
    echo 'Using the vscode terminal, you can make Ctrl-D detach rather than'
    echo 'exit the tmux session if you add add this keybinding to'
    echo 'keybindings.json:'
    echo ''
    echo '{'
    echo '    "key": "ctrl+d",'
    echo '    "command": "workbench.action.terminal.kill", '
    echo '    "when": "terminalFocus"'
    echo '}'
    echo ''
    echo "Commands in session:"
    echo "  detach       Detach session (keeps running)"
    echo "  exit         Terminate session completely"
    echo "  Ctrl+D       Close terminal (if mapped in VS Code)"
    echo ""
    echo "Features:"
    echo "  - Auto-detaches when terminal closes"
    echo "  - Creates session if it doesn't exist"
    echo "  - Attaches to existing session if found"
    echo "  - Session name based on directory basename"
    echo "  - Preserves sessions across terminal closes"
}

# Handle help flag
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Set up trap to catch terminal close/exit signals
trap cleanup EXIT INT TERM

# Record this wrapper instance
echo "$$:$(date)" > "$LOCK_FILE"

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    print_error "tmux is not installed or not in PATH"
    exit 1
fi

# Show session info
print_status "Target directory: $PWD"
print_status "Session name: $SESSION_NAME"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    print_success "Attaching to existing session: $SESSION_NAME"
else
    print_success "Creating new session: $SESSION_NAME"
fi

print_warning "Session will auto-detach when terminal closes"
print_status "Use 'detach' command to disconnect manually"
echo ""

# Start or attach to tmux session
if tmux new-session -A -s "$SESSION_NAME" -c "$PWD" \; \
    send-keys "export PS1='[$SESSION_NAME] \\w $ '" Enter \; \
    send-keys "alias detach='echo Detaching...; tmux detach-client'" Enter \; \
    send-keys "alias det='detach'" Enter \; \
    send-keys "[[ -f ~/.bashrc ]] && source ~/.bashrc" Enter \; \
    send-keys "export PS1='[$SESSION_NAME] \\w $ '" Enter \; \
    send-keys "echo 'Session: $SESSION_NAME | Directory: $PWD'" Enter \; \
    send-keys "echo 'Commands: detach (disconnect) | exit (terminate)'" Enter \; \
    send-keys "clear" Enter; then
    print_success "Session completed"
else
    print_error "Session failed"
    exit 1
fi