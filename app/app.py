from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import os
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient, ASCENDING, DESCENDING
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
from utils.chat_utils import format_messages_for_display
from flask import send_from_directory
from bson import ObjectId
import time
from utils.chat_utils import (
    format_timestamp,
    parse_timestamp,
    format_message_for_socket,
    format_messages_for_display,
    validate_message
)

load_dotenv()

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = MongoClient(os.getenv("MONGO_URI"))
    client.admin.command('ping')
    db = client["skill_exchange"]
    users_col = db["users"]
    messages_col = db["messages"]
    sessions_col = db["sessions"]
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise

def create_indexes():
    users_col.create_index([("username", ASCENDING)], unique=True)
    users_col.create_index([("connections", ASCENDING)])
    messages_col.create_index([("room", ASCENDING), ("timestamp", ASCENDING)])
    sessions_col.create_index([("teacher", ASCENDING)])
    sessions_col.create_index([("learner", ASCENDING)])
    sessions_col.create_index([("status", ASCENDING)])
    sessions_col.create_index([("skill", ASCENDING)])  

create_indexes()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
socketio = SocketIO(app, async_mode='threading')


STARTING_CREDITS = 10
TEACHING_CREDITS = 5
LEARNING_CREDITS = 5


SKILL_BASE_COST = 5
SKILL_DEMAND_MULTIPLIER = 0.5  

def calculate_skill_cost(skill):
    demand_count = users_col.count_documents({'wants': skill})
    return SKILL_BASE_COST + (demand_count * SKILL_DEMAND_MULTIPLIER)

def update_user_skill_stats(username, skill, is_teaching=True):
    update_field = f"teaching_stats.{skill}" if is_teaching else f"learning_stats.{skill}"
    users_col.update_one(
        {'username': username},
        {
            '$inc': {
                f"{update_field}.hours": 1,
                f"{update_field}.sessions": 1
            },
            '$setOnInsert': {
                f"{update_field}.last_activity": datetime.now()
            }
        }
    )

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/transactions')
def transactions():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user = users_col.find_one({'username': username})
    
    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))
    
    if 'transactions' not in user:
        user['transactions'] = []
    
    return render_template('transactions.html', transactions=user.get('transactions', []))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if users_col.find_one({'username': username}):
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user_data = {
            'username': username,
            'password': generate_password_hash(password),
            'offers': [],
            'wants': [],
            'bio': '',
            'connections': [],
            'credits': STARTING_CREDITS,
            'transactions': [],
            'teaching_stats': {},
            'learning_stats': {},
            'learning_streak': 0,
            'last_learning_date': None,
            'created_at': datetime.now()
        }
        
        users_col.insert_one(user_data)
        session['username'] = username
        return redirect(url_for('select_skills'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = users_col.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = users_col.find_one({'username': username})
    matches = [...] 

    suggested_users = []
    if 'embedding' in user:
        user_embedding = np.array(user['embedding'])
        connections = user.get('connections', [])
        others = list(users_col.find({
            'username': {'$ne': username, '$nin': connections},
            'embedding': {'$exists': True}
        }))
        for u in others:
            sim = cosine_similarity([user_embedding], [np.array(u['embedding'])])[0][0]
            if sim >= 0.5:
                u['similarity'] = round(sim * 100, 2)
                suggested_users.append(u)
        suggested_users.sort(key=lambda x: x['similarity'], reverse=True)

    return render_template("dashboard.html", user=user, matches=matches, suggested_users=suggested_users)


@app.route('/profile')
def profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user = users_col.find_one({'username': username})
    
    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))

    if 'offers' not in user:
        user['offers'] = []
    if 'wants' not in user:
        user['wants'] = []
    if 'bio' not in user:
        user['bio'] = ''
    if 'credits' not in user:
        user['credits'] = STARTING_CREDITS
        users_col.update_one(
            {'username': username},
            {'$set': {'credits': STARTING_CREDITS}}
        )
    
    return render_template('profile.html', user=user)

@app.route('/select_skills', methods=['GET', 'POST'])
def select_skills():
    username = session['username']
    user = users_col.find_one({'username': username})

    if request.method == 'POST':
        bio = request.form.get('bio', '')
        offers = request.form.getlist('offers')
        wants = request.form.getlist('wants')
        offers_text = ' '.join(offers)
        wants_text = ' '.join(wants)
        combined_text = offers_text + ' ' + wants_text
        embedding = model.encode(combined_text).tolist()
        users_col.update_one({'username': username}, {
            '$set': {
                'bio': bio,
                'offers': offers,
                'wants': wants,
                'embedding': embedding
            }
        })
        flash('Skills saved successfully!', 'success')
        return redirect(url_for('dashboard'))  # or wherever you want to go after

    return render_template('select_skills.html', user=user)

@app.route('/matches')
def matches():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = users_col.find_one({'username': username})
    if not user or 'embedding' not in user:
        flash('Please update your skills to view matches.')
        return redirect(url_for('select_skills'))

    user_embedding = np.array(user['embedding'])
    connections = user.get('connections', [])
    pending = user.get('pending_requests', [])

    # Fetch users connected with this user
    connected_users = users_col.find({'username': {'$in': connections}})
    connected_list = []
    for cu in connected_users:
        if 'embedding' in cu:
            sim = cosine_similarity([user_embedding], [np.array(cu['embedding'])])[0][0]
            cu['similarity'] = round(sim * 100, 2)
        connected_list.append(cu)

    # Fetch users you sent requests to
    pending_users = users_col.find({'username': {'$in': pending}})
    pending_list = []
    for pu in pending_users:
        if 'embedding' in pu:
            sim = cosine_similarity([user_embedding], [np.array(pu['embedding'])])[0][0]
            pu['similarity'] = round(sim * 100, 2)
        pending_list.append(pu)

    # Suggested users are the rest (can reuse logic from `/suggested`)
    suggestions = []
    all_users = users_col.find({
        'username': {'$ne': username, '$nin': connections + pending},
        'embedding': {'$exists': True}
    })

    for other in all_users:
        sim = cosine_similarity([user_embedding], [np.array(other['embedding'])])[0][0]
        if sim >= 0.5:
            other['similarity'] = round(sim * 100, 2)
            suggestions.append(other)
    suggestions.sort(key=lambda x: x['similarity'], reverse=True)

    return render_template("matches.html",
                           connected=connected_list,
                           pending=pending_list,
                           suggested=suggestions)


@app.route('/suggested')
def suggested():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = users_col.find_one({'username': username})

    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))

    if not user.get('offers') or not user.get('wants'):
        flash('Please set up your skills to get suggestions.')
        return redirect(url_for('select_skills'))

    user_connections = user.get('connections', [])
    suggestions = []

    user_offers_embedding = model.encode(" ".join(user['offers']))
    user_wants_embedding = model.encode(" ".join(user['wants']))

    other_users = list(users_col.find({
        '$and': [
            {'username': {'$ne': username}},
            {'username': {'$nin': user_connections}},
            {'offers': {'$exists': True, '$ne': []}},
            {'wants': {'$exists': True, '$ne': []}}
        ]
    }))

    for other_user in other_users:
        other_offers_embedding = model.encode(" ".join(other_user['offers']))
        other_wants_embedding = model.encode(" ".join(other_user['wants']))

        teach_score = cosine_similarity([user_wants_embedding], [other_offers_embedding])[0][0]
        learn_score = cosine_similarity([user_offers_embedding], [other_wants_embedding])[0][0]

        if teach_score >= 0.5 and learn_score >= 0.5:
            other_user['match_reasons'] = []

            if teach_score >= 0.5:
                other_user['match_reasons'].append("They can teach what you want to learn")
            if learn_score >= 0.5:
                other_user['match_reasons'].append("They want to learn what you can teach")

            other_user['similarity'] = round(((teach_score + learn_score) / 2) * 100, 2)
            suggestions.append(other_user)

    suggestions.sort(key=lambda x: x['similarity'], reverse=True)
    return render_template('suggested.html', suggested_users=suggestions)



@app.route('/send_request/<target_username>', methods=['POST'])
def send_request(target_username):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    target_user = users_col.find_one({'username': target_username})
    
    if target_user:
        users_col.update_one(
            {'username': username},
            {'$addToSet': {'connections': target_username}}
        )
        users_col.update_one(
            {'username': target_username},
            {'$addToSet': {'connections': username}}
        )
    
    return redirect(url_for('suggested'))

@app.route('/messages')
def messages():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    user = users_col.find_one({'username': username})

    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))

    connections_data = []
    for connected_user_username in user.get('connections', []):
        other_user_data = users_col.find_one({'username': connected_user_username})
        if other_user_data:
            room = '_'.join(sorted([username, connected_user_username]))
            latest_message = messages_col.find({'room': room}).sort('timestamp', -1).limit(1)
            latest_message_doc = next(latest_message, None)

            latest_message_for_display = None
            if latest_message_doc:
                message_text = latest_message_doc.get('content') or latest_message_doc.get('message', '')
                timestamp_raw_latest = latest_message_doc.get('timestamp', datetime.now())
                timestamp_formatted = format_timestamp(timestamp_raw_latest)
                latest_message_for_display = {
                    'message': message_text,
                    'timestamp': timestamp_formatted,
                    'raw_timestamp': timestamp_raw_latest
                }
            connections_data.append({
                'username': connected_user_username,
                'latest_message': latest_message_for_display,
                'offers': other_user_data.get('offers', []),
                'wants': other_user_data.get('wants', []),
                'bio': other_user_data.get('bio', '')
            })
    connections_data.sort(key=lambda x: x['latest_message']['raw_timestamp'] if x['latest_message'] else datetime.min, reverse=True)

    if connections_data:
        first_chat_partner = connections_data[0]['username']
        return redirect(url_for('chat', other_user=first_chat_partner))
    else:
        return render_template(
            'chat.html',
            current_user=user,
            connections=[],
            messages=[],
            other_user=None,
            other_user_data=None
        )

@app.route('/chat/<other_user>', methods=['GET', 'POST'])
def chat(other_user):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user = users_col.find_one({'username': username})
    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))

    if other_user not in user.get('connections', []):
        flash('You must be connected to chat with this user.')
        return redirect(url_for('matches'))

    room = '_'.join(sorted([username, other_user]))

    if request.method == 'POST':
        message_text = request.form.get('message')
        if message_text:
            messages_col.insert_one({
                'room': room,
                'sender': username,
                'receiver': other_user,
                'content': message_text,
                'timestamp': datetime.utcnow()
            })
        return redirect(url_for('chat', other_user=other_user))

    chat_history_docs = list(messages_col.find({'room': room}).sort('timestamp', 1))
    formatted_chat_messages = []
    for msg_doc in chat_history_docs:
        message_text = msg_doc.get('content') or msg_doc.get('message', '')
        sender_name = msg_doc.get('sender') or msg_doc.get('username', 'Unknown')
        timestamp_raw = msg_doc.get('timestamp', datetime.now())
        
        formatted_chat_messages.append({
            'sender': sender_name,
            'text': message_text,
            'timestamp': format_timestamp(timestamp_raw)
        })

    connections_data = []
    for connected_user_username in user.get('connections', []):
        other_user_sidebar_data = users_col.find_one({'username': connected_user_username})
        if other_user_sidebar_data:
            sidebar_room = '_'.join(sorted([username, connected_user_username]))
            latest_message = messages_col.find({'room': sidebar_room}).sort('timestamp', -1).limit(1)
            latest_message_doc = next(latest_message, None)

            latest_message_for_display = None
            if latest_message_doc:
                message_text = latest_message_doc.get('content') or latest_message_doc.get('message', '')
                timestamp_raw_latest = latest_message_doc.get('timestamp', datetime.now())
                timestamp_formatted = format_timestamp(timestamp_raw_latest)
                latest_message_for_display = {
                    'message': message_text,
                    'timestamp': timestamp_formatted,
                    'raw_timestamp': timestamp_raw_latest
                }

            connections_data.append({
                'username': connected_user_username,
                'latest_message': latest_message_for_display,
                'offers': other_user_sidebar_data.get('offers', []),
                'wants': other_user_sidebar_data.get('wants', []),
                'bio': other_user_sidebar_data.get('bio', '')
            })
    
    connections_data.sort(key=lambda x: x['latest_message']['raw_timestamp'] if x['latest_message'] else datetime.min, reverse=True)

    other_user_data = users_col.find_one({'username': other_user})

    return render_template('chat.html',
                           current_user=user,
                           other_user=other_user,
                           other_user_data=other_user_data,
                           messages=formatted_chat_messages,
                           connections=connections_data)

@app.route('/clear_chat/<other_user>', methods=['POST'])
def clear_chat(other_user):
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    room = '_'.join(sorted([username, other_user]))

    try:
        messages_col.delete_many({'room': room})
        flash(f"Chat with {other_user} cleared.")
    except Exception as e:
        print(f"Error clearing chat: {e}")
        flash("Failed to clear chat.")

    return redirect(url_for('chat', other_user=other_user))


@socketio.on('join')
def on_join(data):
    username = session['username']
    other_user = data['other_user']
    room = '_'.join(sorted([username, other_user]))
    join_room(room)
    emit('status', {'msg': f'{username} has joined the chat.'}, room=room)

@socketio.on('leave')
def on_leave(data):
    username = session['username']
    other_user = data['other_user']
    room = '_'.join(sorted([username, other_user]))
    leave_room(room)
    emit('status', {'msg': f'{username} has left the chat.'}, room=room)

@socketio.on('typing')
def handle_typing(data):
    if 'username' not in session:
        return
    username = session['username']
    other_user = data.get('other_user')
    room = '_'.join(sorted([username, other_user]))
    emit('typing', {'username': username}, room=room)


@socketio.on('message')
def handle_message(data):
    print(f"handle_message received data: {data}") 
    if 'username' not in session:
        emit('error', {'message': 'Not authenticated'})
        print("Authentication error: User not in session")
        return
    username = session['username']
    other_user = data.get('other_user')
    if not other_user:
        emit('error', {'message': 'Invalid recipient'})
        print("Invalid recipient: other_user not found")
        return
    room = '_'.join(sorted([username, other_user]))
    message_content = data.get('message', '').strip()
    current_timestamp = datetime.now()
    message = {
        'username': username,
        'content': message_content, 
        'timestamp': current_timestamp,
        'room': room
    }
    validation_error = validate_message(message)
    if validation_error:
        emit('error', {'message': validation_error})
        print(f"Validation error: {validation_error}")
        return
    
    try:
        messages_col.insert_one(message)
        formatted_message_for_socket = {
            'username': message.get('username', 'Unknown'),
            'message': message.get('content', ''),
            'timestamp': format_timestamp(message.get('timestamp', datetime.now())),
            'room': message.get('room', '')
        }
        emit('message', formatted_message_for_socket, room=room)
        print(f"Successfully emitted message: {formatted_message_for_socket}")
    except Exception as e:
        print(f"Error handling message: {e}")
        emit('error', {'message': 'Failed to send message'}, room=room)

@app.route('/sessions')
def sessions():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user = users_col.find_one({'username': username})
    
    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))
    

    connections = list(users_col.find({'username': {'$in': user.get('connections', [])}}))

    user_sessions = list(sessions_col.find({
        '$or': [
            {'teacher': username},
            {'learner': username}
        ]
    }).sort('start_time', 1))
    
    return render_template('sessions.html', sessions=user_sessions, connections=connections, user=username)

@app.route('/create_session', methods=['POST'])
def create_session():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    session_type = request.form.get('session_type')
    partner = request.form.get('partner')
    skill = request.form.get('skill')
    objectives = request.form.get('objectives')
    start_time = datetime.fromisoformat(request.form.get('start_time').replace('Z', '+00:00'))
    duration_minutes = int(request.form.get('duration_minutes'))
    if session_type == 'teaching':
        teacher = username
        learner = partner
    else:  
        teacher = partner
        learner = username

    user = users_col.find_one({'username': username})
    if partner not in user.get('connections', []):
        flash('Invalid partner selected')
        return redirect(url_for('sessions'))
    learner_user = users_col.find_one({'username': learner})
    if learner_user.get('credits', 0) < LEARNING_CREDITS:
        flash('Not enough credits to schedule this session')
        return redirect(url_for('sessions'))
    session_data = {
        'teacher': teacher,
        'learner': learner,
        'skill': skill,
        'objectives': objectives,
        'start_time': start_time,
        'duration_minutes': duration_minutes,
        'status': 'scheduled',
        'created_at': datetime.now()
    }
    try:
        sessions_col.insert_one(session_data)
        flash('Session scheduled successfully!')
    except Exception as e:
        print(f"Error creating session: {e}")
        flash('Failed to schedule session')
    
    return redirect(url_for('sessions'))

@app.route('/complete_session/<session_id>', methods=['POST'])
def complete_session(session_id):
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    session_data = sessions_col.find_one({'_id': ObjectId(session_id)})
    
    if not session_data:
        flash('Session not found')
        return redirect(url_for('sessions'))
    teacher = session_data['teacher']
    learner = session_data['learner']
    skill = session_data['skill']
    skill_cost = calculate_skill_cost(skill)
    users_col.update_one(
        {'username': teacher},
        {'$inc': {'credits': skill_cost}}
    )
    users_col.update_one(
        {'username': learner},
        {'$inc': {'credits': -skill_cost}}
    )
    update_user_skill_stats(teacher, skill, is_teaching=True)
    update_user_skill_stats(learner, skill, is_teaching=False)
    learner_user = users_col.find_one({'username': learner})
    last_learning_date = learner_user.get('last_learning_date')
    current_streak = learner_user.get('learning_streak', 0)
    if last_learning_date:
        days_since_last = (datetime.now() - last_learning_date).days
        if days_since_last == 1:
            new_streak = current_streak + 1
        elif days_since_last > 1:
            new_streak = 1
        else: 
            new_streak = current_streak
    else:
        new_streak = 1
    timestamp = datetime.now()
    teacher_transaction = {
        'type': 'earn',
        'amount': skill_cost,
        'description': f'Earned {skill_cost} credits from teaching {learner} {skill}',
        'timestamp': timestamp,
        'skill': skill
    }
    learner_transaction = {
        'type': 'spend',
        'amount': -skill_cost,
        'description': f'Spent {skill_cost} credits on learning {skill} from {teacher}',
        'timestamp': timestamp,
        'skill': skill
    }
    users_col.update_one(
        {'username': teacher},
        {'$push': {'transactions': teacher_transaction}}
    )
    users_col.update_one(
        {'username': learner},
        {
            '$push': {'transactions': learner_transaction},
            '$set': {
                'learning_streak': new_streak,
                'last_learning_date': timestamp
            }
        }
    )
    sessions_col.update_one(
        {'_id': ObjectId(session_id)},
        {'$set': {'status': 'completed'}}
    )
    flash(f'Session completed! {skill_cost} credits have been exchanged.')
    return redirect(url_for('sessions'))

@app.context_processor
def inject_credits():
    if 'username' in session:
        username = session['username']
        user = users_col.find_one({'username': username})
        if user and 'credits' in user:
            return {'user_credits': user['credits']}
    return {'user_credits': 0}

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user = users_col.find_one({'username': username})
    
    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        bio = request.form.get('bio', '').strip()
        users_col.update_one(
            {'username': username},
            {'$set': {'bio': bio}}
        )
        flash('Profile updated successfully!')
        return redirect(url_for('profile'))
    
    return render_template('edit_profile.html', user=user)

@app.route('/edit_skills', methods=['GET', 'POST'])
def edit_skills():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    username = session['username']
    user = users_col.find_one({'username': username})
    
    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        offers = request.form.getlist('offers')
        wants = request.form.getlist('wants')
        
        users_col.update_one(
            {'username': username},
            {'$set': {
                'offers': offers,
                'wants': wants
            }}
        )
        flash('Skills updated successfully!')
        return redirect(url_for('profile'))
    

    all_skills = [

        'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go',
        'HTML/CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Django', 'Flask', 'Spring Boot',
        'SQL', 'MongoDB', 'AWS', 'Docker', 'Kubernetes', 'Git',

        'UI/UX Design', 'Graphic Design', 'Web Design', 'Illustration', 'Digital Art',
        '3D Modeling', 'Animation', 'Video Editing', 'Photography', 'Photo Editing',

        'Project Management', 'Business Analysis', 'Data Analysis', 'Excel', 'Power BI',

        'English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese', 'Korean',
        'Portuguese', 'Italian', 'Russian',

        'Piano', 'Guitar', 'Violin', 'Drums', 'Singing', 'Music Theory', 'Composition',
        'Drawing', 'Painting', 'Sculpture',

        'Cooking', 'Baking', 'Gardening', 'Fitness', 'Yoga', 'Meditation',
        'Time Management', 'Personal Finance', 'Leadership', 'Communication'
    ]
    
    return render_template('edit_skills.html', 
                         user=user, 
                         all_skills=all_skills,
                         selected_offers=user.get('offers', []),
                         selected_wants=user.get('wants', []))

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    user = users_col.find_one({'username': username})
    if not user:
        session.pop('username', None)
        flash('User not found. Please log in again.')
        return redirect(url_for('login'))

    feedback_users = list(users_col.find({'username': {'$ne': username}}, {'username': 1}))

    if request.method == 'POST':
        target_user = request.form['target_user']
        rating = int(request.form['rating'])
        comment = request.form['comment']
        feedback_entry = {
            'from_user': username,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now()
        }
        users_col.update_one({'username': target_user}, {'$push': {'feedback': feedback_entry}})
        flash('Feedback submitted!')
        return redirect(url_for('feedback'))

    feedback_received = user.get('feedback', [])
    feedback_received = sorted(feedback_received, key=lambda x: x['timestamp'], reverse=True)
    return render_template('feedback.html', feedback_users=feedback_users, feedback_received=feedback_received)

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Registered routes:", app.url_map)
    socketio.run(app, debug=True) 