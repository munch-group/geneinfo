#!/bin/bash

# Enhanced Claude Session Wrapper
# Maintains persistent UUID sessions per directory with additional features

UUID_FILE=".claude-uuid"
VERBOSE=false

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -n|--new)
            echo "🗑️  Removing existing session..."
            rm -f "$UUID_FILE"
            shift
            ;;
        -s|--show)
            if [ -f "$UUID_FILE" ]; then
                uuid=$(cat "$UUID_FILE")
                echo "📋 Current session UUID: $uuid"
                echo "📁 Directory: $(pwd)"
                echo "📄 UUID file: $UUID_FILE"
            else
                echo "No session found in current directory"
            fi
            exit 0
            ;;
        -h|--help)
            echo "Claude Session Wrapper"
            echo ""
            echo "Usage: $0 [options] [claude arguments]"
            echo ""
            echo "Options:"
            echo "  -n, --new      Create new session (removes existing UUID)"
            echo "  -s, --show     Show current session info"
            echo "  -v, --verbose  Verbose output"
            echo "  -h, --help     Show this help"
            echo ""
            echo "Examples:"
            echo "  $0              # Start/resume session"
            echo "  $0 --new        # Force new session"
            echo "  $0 --show       # Show session info"
            echo "  $0 'help me'    # Pass arguments to Claude"
            exit 0
            ;;
        *)
            # Remaining arguments go to Claude
            break
            ;;
    esac
done

# Function to generate UUID (cross-platform)
generate_uuid() {
    if command -v uuidgen >/dev/null 2>&1; then
        uuidgen
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import uuid; print(uuid.uuid4())"
    elif command -v python >/dev/null 2>&1; then
        python -c "import uuid; print(uuid.uuid4())"
    else
        # Fallback: generate a pseudo-UUID
        echo "$(date +%s)-$(shuf -i 1000-9999 -n 1 2>/dev/null || echo $RANDOM)-$(shuf -i 1000-9999 -n 1 2>/dev/null || echo $RANDOM)-$(shuf -i 1000-9999 -n 1 2>/dev/null || echo $RANDOM)"
    fi
}

# Function for verbose logging
log() {
    if [ "$VERBOSE" = true ]; then
        echo "🔍 $1"
    fi
}

# Check if Claude is available
if ! command -v claude >/dev/null 2>&1; then
    echo "Error: 'claude' command not found in PATH"
    echo "Please install Claude Code CLI first"
    exit 1
fi

# Get directory name for display
DIR_NAME=$(basename "$PWD")

# Check if UUID file exists
if [ -f "$UUID_FILE" ]; then
    # Read existing UUID
    SESSION_UUID=$(cat "$UUID_FILE")
    
    # Validate UUID format (basic check)
    if [[ ! "$SESSION_UUID" =~ ^[0-9a-fA-F-]+$ ]]; then
        echo "Invalid UUID in $UUID_FILE, generating new one..."
        SESSION_UUID=$(generate_uuid)
        echo "$SESSION_UUID" > "$UUID_FILE"
    else
        echo "📂 Resuming session for '$DIR_NAME'"
        log "UUID: $SESSION_UUID"
        log "File: $UUID_FILE"
    fi
else
    # Generate new UUID and save it
    SESSION_UUID=$(generate_uuid)
    echo "$SESSION_UUID" > "$UUID_FILE"
    echo "Created new session for '$DIR_NAME'"
    log "UUID: $SESSION_UUID"
    log "Saved to: $UUID_FILE"
fi

# Add UUID file to .gitignore if git repo exists
if [ -d ".git" ] && ! grep -q "\.claude-uuid" .gitignore 2>/dev/null; then
    echo ".claude-uuid" >> .gitignore
    log "Added .claude-uuid to .gitignore"
fi

# Start Claude with the UUID and any additional arguments
echo "Starting Claude session..."
log "Command: claude --session-id '$SESSION_UUID' $*"

claude --session-id "$SESSION_UUID" "$@"