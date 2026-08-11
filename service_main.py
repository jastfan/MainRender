from flask import Flask
from RenderDetect import init_render
from hook_detector2 import hooks_bp     # <-- naya import

app = Flask(__name__)
init_render(app)
app.register_blueprint(hooks_bp)        # <-- naya register

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
