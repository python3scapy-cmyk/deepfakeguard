from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # loosen CORS for local dev only

# --- existing score-collection endpoints stay as they were ---
latest_scores = {}

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/score', methods=['POST'])
def receive_score():
    payload = request.get_json()
    module = payload.get("module")
    latest_scores[module] = payload
    return jsonify({"received": True})

@app.route('/scores', methods=['GET'])
def get_scores():
    return jsonify(latest_scores)

# --- new: WebRTC signaling ---
# Simple 2-person room: whoever joins "demo-room" first waits,
# second person triggers the offer/answer exchange.

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    emit('peer-joined', {}, room=room, include_self=False)

@socketio.on('offer')
def on_offer(data):
    emit('offer', data, room=data['room'], include_self=False)

@socketio.on('answer')
def on_answer(data):
    emit('answer', data, room=data['room'], include_self=False)

@socketio.on('ice-candidate')
def on_ice_candidate(data):
    emit('ice-candidate', data, room=data['room'], include_self=False)

if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True)