#!/bin/sh
echo "Starting Ollama server in the background..."
# Start the server in the background and record its PID
ollama serve &
OLLAMA_PID=$!

# Pull required models now that the server is available
echo "Pulling model smollm:135m..."
ollama pull smollm:135m || { echo "Failed to pull smollm:135m"; exit 1; }

echo "Pulling model deepseek-r1:14b..."
ollama pull deepseek-r1:14b || { echo "Failed to pull deepseek-r1:14b"; exit 1; }

echo "Pulling model qwen2.5:14b..."
ollama pull qwen2.5:14b || { echo "Failed to pull qwen2.5:14b"; exit 1; }

echo "Pulling summarization model phi3:mini..."
ollama pull phi3:mini || { echo "Failed to pull phi3:mini"; exit 1; }

echo "All models pulled successfully. Monitoring Ollama server..."
# Wait on the Ollama server process so the container stays alive
wait $OLLAMA_PID
