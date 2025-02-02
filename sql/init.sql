-- Drop Existing Tables
DROP TABLE IF EXISTS quiz_responses;
DROP TABLE IF EXISTS quiz_questions;
DROP TABLE IF EXISTS chatbot_feedback;
DROP TABLE IF EXISTS chatbot_history;
DROP TABLE IF EXISTS chat_sessions;
DROP TABLE IF EXISTS users;

-- Users Table
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Chat Sessions
CREATE TABLE chat_sessions (
    chat_title TEXT PRIMARY KEY,
    username TEXT REFERENCES users(username) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chatbot History
CREATE TABLE chatbot_history (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_title TEXT REFERENCES chat_sessions(chat_title) ON DELETE CASCADE,
    username TEXT REFERENCES users(username) ON DELETE CASCADE,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding_model TEXT,
    generative_model TEXT,
    context TEXT,
    safe BOOLEAN DEFAULT TRUE,
    latency REAL,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    query_price REAL,
    energy_usage REAL,
    gwp REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- LLM Logs for Quiz Queries
CREATE TABLE llm_logs_quiz (
    log_id SERIAL PRIMARY KEY,
    username TEXT REFERENCES users(username) ON DELETE CASCADE,
    query TEXT NOT NULL,
    response TEXT,
    generative_model TEXT NOT NULL,
    energy_usage REAL,
    gwp REAL,
    completion_tokens INT,
    prompt_tokens INT,
    query_price REAL,
    execution_time_ms FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Feedback Table
CREATE TABLE chatbot_feedback (
    id SERIAL PRIMARY KEY,
    query_id UUID REFERENCES chatbot_history(query_id) ON DELETE CASCADE,
    username TEXT REFERENCES users(username) ON DELETE CASCADE,
    feedback TEXT CHECK (feedback IN ('Utile', 'Inutile')) NOT NULL,
    comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quiz Questions
CREATE TABLE quiz_questions (
    quiz_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT REFERENCES users(username) ON DELETE CASCADE,
    question TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Quiz Responses
CREATE TABLE quiz_responses (
    response_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quiz_id UUID REFERENCES quiz_questions(quiz_id) ON DELETE CASCADE,
    username TEXT REFERENCES users(username) ON DELETE CASCADE,
    user_answer TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for Performance Optimization
CREATE INDEX idx_chatbot_history_username ON chatbot_history(username);
CREATE INDEX idx_chatbot_history_timestamp ON chatbot_history(timestamp);
CREATE INDEX idx_chatbot_feedback_query_id ON chatbot_feedback(query_id);
CREATE INDEX idx_llm_logs_quiz_username ON llm_logs_quiz(username);

-- Default Users
INSERT INTO users (username, password_hash, role) 
VALUES ('admin', md5('admin123'), 'admin'),
       ('user', md5('user123'), 'user');

-- Default Chat Sessions
INSERT INTO chat_sessions (chat_title, username) VALUES 
    ('Default Chat', 'user'),
    ('Default Chat 2', 'user');

-- Sample Chatbot History
INSERT INTO chatbot_history (chat_title, username, query, answer)
VALUES 
    ('Default Chat', 'user', 'Hello', 'Hi! How can I help you?'),
    ('Default Chat', 'user', 'How are you?', 'I am doing great!'),
    ('Default Chat 2', 'user', 'What is the weather today?', 'The weather is sunny today.'),
    ('Default Chat 2', 'user', 'What is the weather tomorrow?', 'The weather will be rainy tomorrow.');
