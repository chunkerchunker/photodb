from flask import Flask
from pathlib import Path
import os


def create_app(config=None):
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
    app.config["DATABASE_URL"] = os.environ.get("DATABASE_URL", "postgresql://localhost/photodb")

    if config:
        app.config.update(config)

    from .routes import bp

    app.register_blueprint(bp)

    return app
