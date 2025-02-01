-- Drop existing tables if they exist
DROP TABLE IF EXISTS chatbot_history;
DROP TABLE IF EXISTS chat_sessions;
DROP TABLE IF EXISTS users;

-- Create users table
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Create chat_sessions table
CREATE TABLE chat_sessions (
    chat_title TEXT PRIMARY KEY,
    username TEXT REFERENCES users(username),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create chatbot_history table
CREATE TABLE chatbot_history (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_title TEXT REFERENCES chat_sessions(chat_title),
    username TEXT REFERENCES users(username),
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding_model TEXT,
    generative_model TEXT,
    context TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    safe BOOLEAN DEFAULT true,
    latency REAL,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    query_price REAL,
    energy_usage REAL,
    gwp REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_chatbot_history_chat_title ON chatbot_history(chat_title);
CREATE INDEX idx_chatbot_history_username ON chatbot_history(username);
CREATE INDEX idx_chatbot_history_timestamp ON chatbot_history(timestamp);
CREATE INDEX idx_users_username ON users(username);

-- Default Admin User
INSERT INTO users (username, password_hash, role) 
VALUES ('admin', md5('admin123'), 'admin');

-- Default Regular User
INSERT INTO users (username, password_hash, role) 
VALUES ('user', md5('user123'), 'user');

INSERT INTO chat_sessions (chat_title, username)
VALUES ('Default Chat', 'user');

INSERT INTO chat_sessions (chat_title, username)
VALUES ('Default Chat 2', 'user');


INSERT INTO chatbot_history (chat_title, username, query, answer)
VALUES ('Default Chat', 'user', 'Hello', 'Hi! How can I help you?');

INSERT INTO chatbot_history (chat_title, username, query, answer)
VALUES ('Default Chat', 'user', 'How are you?', 'I am doing great!');

INSERT INTO chatbot_history (chat_title, username, query, answer)
VALUES ('Default Chat 2', 'user', 'What is the weather today?', 'The weather is sunny today.');

INSERT INTO chatbot_history (chat_title, username, query, answer)
VALUES ('Default Chat 2', 'user', 'What is the weather tomorrow?', 'The weather will be rainy tomorrow.');
