import sqlite3
from passlib.hash import pbkdf2_sha256
from datetime import datetime

def create_users_db():
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_messages_db():
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id TEXT,
            role TEXT,
            message TEXT,
            user_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES Users(user_id)
        )
    """)
    conn.commit()
    conn.close()


def write_to_messages_db(thread_id, role, message):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO Messages (thread_id, role, message)
        VALUES (?, ?, ?)
    """
    cursor.execute(insert_query, (thread_id, role, message))
    conn.commit()
    conn.close()


def get_all_thread_messages(thread_id):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT message 
                      FROM Messages
                      WHERE thread_id = ?
                      ORDER BY message_id''', (thread_id,))
    rows = cursor.fetchall()
    conn.close()
    return [i[0] for i in rows]


def get_unique_thread_ids():
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT thread_id FROM Messages")
    unique_thread_ids = reversed([row[0] for row in cursor.fetchall()])
    conn.close()
    return unique_thread_ids


def create_documents_db():
    pass


def get_uploaded_doc_names():
    pass


def add_user_to_db(email, password):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    hashed_password = pbkdf2_sha256.hash(password)

    insert_query = """
        INSERT INTO Users (email, password)
        VALUES (?, ?)
    """
    cursor.execute(insert_query, (email, hashed_password))
    conn.commit()
    create_log("Registration", email)
    conn.close()



def authenticate_user(email, password):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    select_query = """
        SELECT password FROM Users WHERE email = ?
    """
    cursor.execute(select_query, (email,))
    user = cursor.fetchone()
    conn.close()
    if user:
        hashed_password = user[0]
        return pbkdf2_sha256.verify(password, hashed_password)
    return False

def create_log_table():
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_registration_log
        (id INTEGER PRIMARY KEY,
        event_time TIMESTAMP, 
        event_type TEXT, 
        username TEXT)
    """)
    conn.commit()
    conn.close()

def create_log(event_type, username):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()


    insert_query = """INSERT INTO login_registration_log (event_time, event_type, username) VALUES (?, ?, ?)
    """
    cursor.execute(insert_query, (datetime.now(), event_type, username))
    conn.commit()

    conn.close()


