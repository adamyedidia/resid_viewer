from flask import Flask, request, jsonify
import numpy as np
from database import SessionLocal
from model import Model
from resid import Resid
from sqlalchemy import func
from direction import Direction
from flask_cors import CORS
from utils import get_layer_num_from_resid_type

from settings import DATABASE_URL

app = Flask(__name__)
CORS(app)



@app.route('/api/resids', methods=['GET'])
def get_resids():
    sess = SessionLocal()

    model_name = request.args.get('model_name')
    type = request.args.get('type')
    head = request.args.get('head') or None
    component_index = request.args.get('component_index')

    if not type:
        return jsonify([])

    model = (
        sess.query(Model)
        .filter(Model.name == model_name)
        .one_or_none()
    )

    print(model)

    layer = get_layer_num_from_resid_type(type)

    resid_prompt_ids = (
        sess.query(Resid.prompt_id)
        .filter(Resid.model == model)
        .filter(Resid.layer == layer)
        .filter(Resid.type == type)
        .filter(Resid.head == head)
        .order_by(func.random())
        .limit(50)
        .all()
    )

    resid_prompt_ids = {t[0] for t in resid_prompt_ids}

    resids = (
        sess.query(Resid)
        .filter(Resid.model == model)
        .filter(Resid.layer == layer)
        .filter(Resid.type == type)
        .filter(Resid.head == head)
        .filter(Resid.prompt_id.in_(resid_prompt_ids))
        .all()
    )

    return jsonify([resid.to_json() for resid in resids])


@app.route('/api/directions', methods=['GET'])
def get_direction():
    sess = SessionLocal()

    model_name = request.args.get('model_name')
    type = request.args.get('type')
    head = request.args.get('head') or None
    component_index = request.args.get('component_index') or None

    if not type:
        return jsonify([])

    layer = get_layer_num_from_resid_type(type)

    model = (
        sess.query(Model)
        .filter(Model.name == model_name)
        .one_or_none()
    )

    direction = (
        sess.query(Direction)
        .filter(Direction.model == model)
        .filter(Direction.layer == layer)
        .filter(Direction.type == type)
        .filter(Direction.head == head)
        .filter(Direction.component_index == component_index)
        .one_or_none()
    )

    return jsonify(direction.to_json()) if direction else None


if __name__ == '__main__':
    app.run(debug=True)