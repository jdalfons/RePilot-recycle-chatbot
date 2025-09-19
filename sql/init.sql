PRAGMA foreign_keys = ON;

-- Drop existing tables
DROP TABLE IF EXISTS quiz_responses;
DROP TABLE IF EXISTS quiz_questions;
DROP TABLE IF EXISTS chatbot_feedback;
DROP TABLE IF EXISTS llm_logs_quiz;
DROP TABLE IF EXISTS chatbot_history;
DROP TABLE IF EXISTS chat_sessions;
DROP TABLE IF EXISTS users;

-- Users table
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    is_active INTEGER DEFAULT 1
);

-- Chat sessions
CREATE TABLE chat_sessions (
    chat_title TEXT PRIMARY KEY,
    username TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

-- Chatbot history
CREATE TABLE chatbot_history (
    query_id TEXT PRIMARY KEY,
    chat_title TEXT,
    username TEXT,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding_model TEXT,
    generative_model TEXT,
    context TEXT,
    safe INTEGER DEFAULT 1,
    latency REAL,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    query_price REAL,
    energy_usage REAL,
    gwp REAL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chat_title) REFERENCES chat_sessions(chat_title) ON DELETE CASCADE,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

-- LLM logs for quiz queries
CREATE TABLE llm_logs_quiz (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    query TEXT NOT NULL,
    response TEXT,
    generative_model TEXT NOT NULL,
    energy_usage REAL,
    gwp REAL,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    query_price REAL,
    execution_time_ms REAL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

-- User feedback
CREATE TABLE chatbot_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    username TEXT,
    feedback TEXT CHECK (feedback IN ('Utile', 'Inutile')) NOT NULL,
    comment TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (query_id) REFERENCES chatbot_history(query_id) ON DELETE CASCADE,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

-- Quiz questions
CREATE TABLE quiz_questions (
    quiz_id TEXT PRIMARY KEY,
    username TEXT,
    question TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

-- Quiz responses
CREATE TABLE quiz_responses (
    response_id TEXT PRIMARY KEY,
    quiz_id TEXT,
    username TEXT,
    user_answer TEXT NOT NULL,
    is_correct INTEGER NOT NULL,
    answered_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (quiz_id) REFERENCES quiz_questions(quiz_id) ON DELETE CASCADE,
    FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
);

-- Default users
INSERT INTO users (username, password_hash, role) VALUES
('admin', '0192023a7bbd73250516f069df18b500', 'admin'),
('admin2', '0192023a7bbd73250516f069df18b500', 'admin'),
('user', '6e71af3b38892f820164d76925e8c050', 'user');

-- Default chat sessions
INSERT INTO chat_sessions (chat_title, username) VALUES
('Default Chat', 'user'),
('Default Chat 2', 'user');

-- Sample chatbot history
INSERT INTO chatbot_history (query_id, chat_title, username, query, answer) VALUES
('1', 'Default Chat', 'user', 'Hello', 'Hi! How can I help you?'),
('2', 'Default Chat', 'user', 'How are you?', 'I am doing great!'),
('3', 'Default Chat 2', 'user', 'What is the weather today?', 'The weather is sunny today.'),
('4', 'Default Chat 2', 'user', 'What is the weather tomorrow?', 'The weather will be rainy tomorrow.');

