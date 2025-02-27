from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import os
import json
from model_comparison import generate_comparative_report
import logging

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return "<h1>Welcome to the LLM Comparison API</h1><p>This API is used by the frontend to generate reports.</p>"

# Mount the reports directory as static files to serve generated HTML reports
REPORTS_DIR = "/reports"
if os.path.isdir(REPORTS_DIR):
    app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# In-memory store of active WebSocket connections (if needed for broadcasting multiple clients)
active_connections = []

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_connections.append(ws)
    try:
        # Receive initial message from client with generation parameters
        data = await ws.receive_text()
        try:
            params = json.loads(data)
        except Exception as e:
            await ws.send_json({"event": "error", "message": "Invalid JSON parameters."})
            return

        query = params.get("query", "")
        models_param = params.get("models", "")
        n = int(params.get("n", 1))
        reference = params.get("answer", "") or ""  # optional reference answer
        # Parse models list (comma-separated string)
        models = [m.strip() for m in models_param.split(",") if m.strip()]

        if not query or not models:
            await ws.send_json({"event": "error", "message": "Query and models must be provided."})
            return

        # Notify start of processing
        await ws.send_json({"event": "status", "message": f"Starting report generation for query: '{query}'"})

        # Generate the full comparative report, passing the WebSocket for progress updates.
        filename = await generate_comparative_report(query, models, n, reference, ws)

        # Once done, inform client of completion and the report filename
        await ws.send_json({"event": "done", "message": "Report generation completed.", "filename": filename})
    except WebSocketDisconnect:
        active_connections.remove(ws)
    except Exception as e:
        await ws.send_json({"event": "error", "message": f"Internal error: {str(e)}"})
        try:
            active_connections.remove(ws)
        except:
            pass
