#!/bin/sh
echo "Starting Ollama server in the background..."
# Start the server in the background and record its PID
/usr/bin/ollama serve &
OLLAMA_PID=$!

# Pull required models now that the server is available
echo "Pulling model smollm:135m..."
/usr/bin/ollama pull smollm:135m || { echo "Failed to pull smollm:135m"; exit 1; }

echo "Pulling model deepseek-r1:1.5b..."
/usr/bin/ollama pull deepseek-r1:1.5b || { echo "Failed to pull deepseek-r1:1.5b"; exit 1; }

echo "Pulling model qwen:1.8b..."
/usr/bin/ollama pull qwen:1.8b || { echo "Failed to pull qwen:1.8b"; exit 1; }

echo "Pulling summarization model phi3:mini..."
/usr/bin/ollama pull phi3:mini || { echo "Failed to pull phi3:mini"; exit 1; }

echo "All models pulled successfully. Monitoring Ollama server..."
# Wait on the Ollama server process so the container stays alive
wait $OLLAMA_PID
