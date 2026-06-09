from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime

app = FastAPI()

# Global state tracker
current_student_state = 0

STATE_LABELS = {0: "FOCUSED", 1: "DISTRACTED", 2: "DROWSY"}


@app.post("/update_state")
async def update_state(request: Request):
    global current_student_state
    data = await request.json()
    new_state = data["new_state"]
    timestamp = data["timestamp"]

    current_student_state = new_state
    readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    label = STATE_LABELS.get(new_state, "UNKNOWN")

    if new_state == 0:
        print(f"✅ [{readable_time}] Student is {label}")
    elif new_state == 1:
        print(f"⚠️  PROCTOR ALERT [{readable_time}]: Student is {label}")
    elif new_state == 2:
        print(f"🚨 PROCTOR ALERT [{readable_time}]: Student is {label}")
    else:
        print(f"❓ [{readable_time}]: Unknown state received ({new_state})")

    return {"status": "ok", "received_state": new_state}


@app.get("/get_state")
async def get_state():
    return {
        "current_state": current_student_state,
        "label": STATE_LABELS.get(current_student_state, "UNKNOWN"),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
