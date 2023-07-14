from functools import wraps
from typing import Optional
from flask import Flask, request, jsonify
import numpy as np
from database import SessionLocal
from model import Model
from resid import Resid
from sqlalchemy import func
from direction import Direction
from flask_cors import CORS
from direction import add_direction
from utils import get_layer_num_from_resid_type
from user import add_or_get_user, User
from direction_description import DirectionDescription

from settings import DATABASE_URL

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def sess_decorator(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        with SessionLocal() as sess:
            kwargs['sess'] = sess
            try:
                response = f(*args, **kwargs)
            except Exception as e:
                sess.rollback()
                return jsonify({'error': str(e)}), 500
            finally:
                sess.close()
        return response
    return decorated_function


def parse_optional_int(val):
    try:
        return int(val)
    except ValueError:
        return None


@app.route('/api/resids', methods=['GET'])
@sess_decorator
def get_resids(sess):
    model_name = request.args.get('model_name')
    type = request.args.get('type')
    head = parse_optional_int(request.args.get('head'))
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
        .filter(Resid.token_position > 0)  # The leading |<endoftext>| token is weird
        .all()
    )

    return jsonify([resid.to_json() for resid in resids])


@app.route('/api/directions', methods=['GET'])
@sess_decorator
def get_pca_direction(sess):
    model_name = request.args.get('model_name')
    type = request.args.get('type')
    head = parse_optional_int(request.args.get('head'))
    component_index = parse_optional_int(request.args.get('component_index'))
    username = request.args.get('username')

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
        .filter(Direction.generated_by_process == 'pca')
        .one_or_none()
    )
    if direction is None:
        return jsonify(None)

    user: Optional[User] = None
    if username:
        user = add_or_get_user(sess, username)

    my_direction_description = (
        sess.query(DirectionDescription)
        .filter(DirectionDescription.user == user)
        .filter(DirectionDescription.direction == direction)
        .one_or_none()
    )

    best_direction_description = (
        sess.query(DirectionDescription)
        .filter(DirectionDescription.direction == direction)
        .order_by(DirectionDescription.upvotes.desc())
        .first()
    )

    return jsonify({
        **direction.to_json(), 
        **({'myDescription': my_direction_description.description} if my_direction_description else {}),
        **({'bestDescription': best_direction_description.description} if best_direction_description else {}),
    })


@app.route('/api/all_directions', methods=['GET'])
@sess_decorator
def get_all_directions(sess):
    model_name = request.args.get('model_name')
    type = request.args.get('type')
    head = parse_optional_int(request.args.get('head'))

    print(type)

    if not type:
        return jsonify([])

    layer = get_layer_num_from_resid_type(type)

    model = (
        sess.query(Model)
        .filter(Model.name == model_name)
        .one_or_none()
    )

    directions = (
        sess.query(Direction)
        .filter(Direction.model == model)
        .filter(Direction.layer == layer)
        .filter(Direction.type == type)
        .filter(Direction.head == head)
        .all()
    )

    return jsonify([direction.to_json() for direction in directions])


@app.route('/api/directions', methods=['POST'])
@sess_decorator
def create_direction(sess):
    # Get the JSON data from the request
    data = request.get_json()
    model_name = data['model_name']
    type = data['type']
    head = parse_optional_int(data['head'])
    direction = data['direction']
    username = data['username']
    direction_name = data['direction_name']
    direction_description = data['direction_description']

    user = add_or_get_user(sess, username)

    if sess.query(Direction).filter_by(name=direction_name).count() > 0:
        return jsonify({'error': 'Direction with that name already exists'}), 400

    direction_obj = add_direction(
        sess=sess,
        direction=direction,
        model=sess.query(Model).filter(Model.name == model_name).one_or_none(),
        layer=get_layer_num_from_resid_type(type),
        type=type,
        head=head,
        generated_by_process='user_linear_combination',
        component_index=None,
        scaler=None,
        fraction_of_variance_explained=None,
        user=user,
        name=direction_name,
    )

    sess.add(DirectionDescription(
        direction=direction_obj,
        user=user,
        description=direction_description,
    ))

    sess.commit()

    return jsonify({
        **direction_obj.to_json(),
        'myDescription': direction_description,
    })


@app.route('/api/directions/<direction_id>/descriptions', methods=['GET'])
@sess_decorator
def get_direction_descriptions(direction_id, sess):
    username = request.args.get('username')
    direction_id = request.args.get('direction_id')

    direction = (
        sess.query(Direction).get(direction_id)
    )

    if not direction:
        return jsonify(None)

    user: Optional[User] = None
    if username:
        user = add_or_get_user(sess, username)

    my_direction_description = (
        sess.query(DirectionDescription)
        .filter(DirectionDescription.user == user)
        .filter(DirectionDescription.direction == direction)
        .one_or_none()
    )

    best_direction_description = (
        sess.query(DirectionDescription)
        .filter(DirectionDescription.direction == direction)
        .order_by(DirectionDescription.upvotes.desc())
        .first()
    )

    return jsonify({
        **direction.to_json(), 
        **({'myDescription': my_direction_description.description} if my_direction_description else {}),
        **({'bestDescription': best_direction_description.description} if best_direction_description else {}),
    })


@app.route('/api/directions/<direction_id>/descriptions', methods=['POST'])
@sess_decorator
def create_direction_description(direction_id, sess):
    # Get the JSON data from the request
    data = request.get_json()
    username = data['username']
    direction_description = data['direction_description']

    direction = (
        sess.query(Direction).get(direction_id)
    )

    if not direction:
        return jsonify(None)

    user = add_or_get_user(sess, username)

    sess.add(DirectionDescription(
        direction=direction,
        user=user,
        description=direction_description,
    ))

    sess.commit()

    return jsonify(direction.to_json())


@app.route('/api/descriptions/<direction_description_id>/upvote', methods=['POST'])
@sess_decorator
def upvote_direction(direction_description_id, sess):
    direction_description = (
        sess.query(DirectionDescription)
        .filter(DirectionDescription.id == direction_description_id)
        .one_or_none()
    )

    if direction_description is None:
        return jsonify(None)

    direction_description.upvotes += 1  # type: ignore

    sess.commit()

    return jsonify(direction_description.to_json())


if __name__ == '__main__':
    app.run(debug=True)