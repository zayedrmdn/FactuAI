from flask import Flask
from flask_cors import CORS
from config import Config
from db import db
from routes.auth import auth_bp
from routes.summarize import summarize_bp
from routes.factcheck import factcheck_bp

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": "*"}})

db.init_app(app)
app.register_blueprint(auth_bp)

with app.app_context():
    db.create_all()

app.register_blueprint(summarize_bp)
app.register_blueprint(factcheck_bp)

if __name__ == "__main__":
    app.run(debug=True)
