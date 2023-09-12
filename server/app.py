import sys
sys.path.append('..')

from functools import wraps
from typing import Optional
from flask import Flask, request, jsonify
import numpy as np
from server.database import SessionLocal
from server.model import Model
from server.resid import Resid
from sqlalchemy import func
from server.direction import Direction
from flask_cors import CORS
from server.direction import add_direction
from server.utils import enc, get_layer_num_from_resid_type
from server.user import add_or_get_user, User
from server.prompt import Prompt
from server.direction_description import DirectionDescription
from server.resid_writer import write_resids_for_prompt
import traceback

from settings import DATABASE_URL

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def sess_decorator(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            with SessionLocal() as sess:
                kwargs['sess'] = sess
                try:
                    response = f(*args, **kwargs)
                except Exception as e:
                    sess.rollback()
                    raise e
                finally:
                    sess.close()
            return response
        except Exception as e:
            print(traceback.print_exc())
            return jsonify({"error": "Unexpected error"}), 500        
    return decorated_function


def parse_optional_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


@app.route('/api/resids', methods=['GET'])
@sess_decorator
def get_resids(sess):
    model_name = request.args.get('model_name')
    type = request.args.get('type')
    head = parse_optional_int(request.args.get('head'))
    component_index = request.args.get('component_index')
    username = request.args.get('username')

    if not type:
        return jsonify([])

    model = (
        sess.query(Model)
        .filter(Model.name == model_name)
        .one_or_none()
    )

    layer = get_layer_num_from_resid_type(type)

    print(f"username = {username}")
    if username:
        user = add_or_get_user(sess, username)
        print(f"user = {user}")

        my_resid_prompt_ids = (
            sess.query(Prompt.id)
            .filter(Prompt.added_by_user_id == user.id)
            .order_by(Prompt.created_at.desc())
            .limit(5)
            .all()
        )
        print(f"my_resid_prompt_ids = {my_resid_prompt_ids}")
    else:
        my_resid_prompt_ids = set()

    resid_prompt_ids = (
        sess.query(Resid.prompt_id)
        .filter(Resid.model == model)
        .filter(Resid.layer == layer)
        .filter(Resid.type == type)
        .filter(Resid.head == head)
        .filter(Resid.dataset == 'openwebtext-10k')
        .order_by(func.random())
        .limit(30)
        .all()
    )

    resid_prompt_ids = {*{t[0] for t in resid_prompt_ids}, *{t[0] for t in my_resid_prompt_ids}}

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
    print(head)
    print(model_name)

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


@app.route('/api/my_directions', methods=['GET'])
@sess_decorator
def get_my_directions(sess):
    username = request.args.get('username')

    if not username:
        return jsonify([])

    user = add_or_get_user(sess, username)

    directions = (
        sess.query(Direction)
        .filter(Direction.user == user)
        .filter(~Direction.deleted)
        .order_by(Direction.created_at.desc())
        .limit(1000)
        .all()
    )

    direction_jsons = []

    for direction in directions:
        my_direction_description = (
            sess.query(DirectionDescription)
            .filter(DirectionDescription.user == user)
            .filter(DirectionDescription.direction == direction)
            .order_by(DirectionDescription.created_at.desc())
            .first()
        )

        best_direction_description = (
            sess.query(DirectionDescription)
            .filter(DirectionDescription.direction == direction)
            .order_by(DirectionDescription.upvotes.desc())
            .first()
        )

        direction_jsons.append(
            {
                **direction.to_json(), 
                **({'myDescription': my_direction_description.to_json() if my_direction_description is not None else None}),
                **({'bestDescription': best_direction_description.to_json() if best_direction_description is not None else None}),
            }
        )

    return jsonify(direction_jsons)


@app.route('/api/directions/<direction_id>', methods=['DELETE'])
@sess_decorator
def delete_direction(direction_id, sess):
    direction = (
        sess.query(Direction)
        .filter(Direction.id == direction_id)
        .one_or_none()
    )

    if direction is None:
        return jsonify({'error': 'Direction not found'}), 404

    direction.deleted = True
    sess.commit()

    return jsonify({'success': True})


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

    direction_description_obj = DirectionDescription(
        direction=direction_obj,
        user=user,
        description=direction_description,
    )

    sess.add(direction_description_obj)

    sess.commit()

    return jsonify({
        **direction_obj.to_json(),
        'myDescription': direction_description_obj.to_json(),
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
        .order_by(DirectionDescription.created_at.desc())
        .first()
    )

    best_direction_description = (
        sess.query(DirectionDescription)
        .filter(DirectionDescription.direction == direction)
        .order_by(DirectionDescription.upvotes.desc())
        .first()
    )

    return jsonify({
        **direction.to_json(), 
        **({'myDescription': my_direction_description.to_json()} if my_direction_description else {}),
        **({'bestDescription': best_direction_description.to_json()} if best_direction_description else {}),
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


@app.route('/api/descriptions/<direction_description_id>/downvote', methods=['POST'])
@sess_decorator
def downvote_direction(direction_description_id, sess):
    direction_description = (
        sess.query(DirectionDescription)
        .filter(DirectionDescription.id == direction_description_id)
        .one_or_none()
    )

    if direction_description is None:
        return jsonify(None)

    direction_description.upvotes -= 1  # type: ignore

    sess.commit()

    return jsonify(direction_description.to_json())


@app.route('/api/prompts', methods=['POST'])
@sess_decorator
def add_prompt(sess):
    data = request.get_json()
    username = data['username']
    prompt = data['prompt']
    model_name = data['model_name']

    model = (
        sess.query(Model)
        .filter(Model.name == model_name)
        .one_or_none()
    )

    if not prompt:
        return jsonify({'error': 'No prompt text provided'}), 400

    if not model:
        return jsonify({'error': f'Unrecognized model {model_name}'}, 400)

    if len(prompt) > 10000:
        return jsonify({'error': 'Prompt too long'}), 400

    user = add_or_get_user(sess, username)

    encoded_text_split_by_token = enc.encode(prompt)
    text_split_by_token = [enc.decode([token]) for token in encoded_text_split_by_token]
    length_in_tokens = len(encoded_text_split_by_token)

    prompt_obj = Prompt(
        text=prompt,
        added_by_user=user,
        encoded_text_split_by_token=encoded_text_split_by_token,
        text_split_by_token=text_split_by_token,
        length_in_tokens=length_in_tokens,
    )

    sess.add(prompt_obj)
    sess.commit()

    write_resids_for_prompt(sess, prompt_obj, model)

    return jsonify({'success': True})


@app.route('/api/prompts', methods=['POST'])
@sess_decorator
def run_intervention(sess):
    data = request.get_json()
    prompt = data['prompt']

    model_name = data['model_name']
    type = data['type']
    head = parse_optional_int(data['head'])
    direction = data['direction']

    username = data['username']

    # TODO: (Andrea) run intervention in direction on the type and head

    return jsonify({'success': True})



if __name__ == '__main__':
    app.run(debug=True)