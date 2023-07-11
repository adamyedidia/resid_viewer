from flask import Flask, request, jsonify
import numpy as np
from database import SessionLocal
from model import Model
from resid import Resid
from sqlalchemy import func

from settings import DATABASE_URL

app = Flask(__name__)


@app.route('/api/resids', methods=['GET'])
def get_resids():
    sess = SessionLocal()

    model_name = request.args.get('model_name')
    layer = request.args.get('layer')
    type = request.args.get('type')
    head = request.args.get('head')
    component_index = request.args.get('component_index')

    model = (
        sess.query(Model)
        .filter(Model.name == model_name)
        .one_or_none()
    )

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


    return jsonify([resid.resid.tolist() for resid in resids])


if __name__ == '__main__':
    app.run(debug=True)