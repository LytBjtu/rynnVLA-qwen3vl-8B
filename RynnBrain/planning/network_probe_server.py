import argparse
import socket
from datetime import datetime, timezone

from fastapi import FastAPI, Request
import uvicorn


app = FastAPI(title="Network Probe Server")


@app.get("/")
def root():
    return {
        "ok": True,
        "service": "network-probe",
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
    }


@app.get("/whoami")
def whoami(request: Request):
    client = request.client
    return {
        "ok": True,
        "client_host": client.host if client else None,
        "client_port": client.port if client else None,
        "server_host": socket.gethostname(),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/healthz")
def healthz():
    return {"status": "healthy"}


def parse_args():
    parser = argparse.ArgumentParser(description="Simple FastAPI server for remote connectivity checks")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=56660, help="Port to bind")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
