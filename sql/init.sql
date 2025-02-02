-- Supprimer les tables existantes si elles existent
DROP TABLE IF EXISTS quiz_responses;
DROP TABLE IF EXISTS quiz_questions;
DROP TABLE IF EXISTS chatbot_feedback;
DROP TABLE IF EXISTS chatbot_history;
DROP TABLE IF EXISTS chat_sessions;
DROP TABLE IF EXISTS users;

-- Table des utilisateurs
CREATE TABLE users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true
);

-- Table des sessions de chat
CREATE TABLE chat_sessions (
    chat_title TEXT PRIMARY KEY,
    username TEXT REFERENCES users(username),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des historiques de conversation
CREATE TABLE chatbot_history (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_title TEXT REFERENCES chat_sessions(chat_title),
    username TEXT REFERENCES users(username),
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    embedding_model TEXT,  -- ✅ Assure-toi que cette ligne est bien là !
    generative_model TEXT,
    context TEXT,
    safe BOOLEAN DEFAULT true,
    latency REAL,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    query_price REAL,
    energy_usage REAL,
    gwp REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE llm_logs_quiz (
    log_id SERIAL PRIMARY KEY,
    username TEXT REFERENCES users(username),  -- 🔹 Associe chaque appel à un utilisateur
    query TEXT NOT NULL,  -- La question envoyée au LLM
    response TEXT,  -- La réponse retournée par le LLM
    generative_model TEXT NOT NULL,  -- Modèle utilisé (ex: "mistral-large")
    safe BOOLEAN DEFAULT true,  -- Si la réponse est jugée sûre
    energy_usage REAL,  -- Consommation d’énergie
    gwp REAL,  -- Potentiel de réchauffement global
    completion_tokens INT,  -- Nombre de tokens générés
    prompt_tokens INT,  -- Nombre de tokens envoyés
    query_price REAL,  -- Prix estimé de la requête
    execution_time_ms FLOAT,  -- Temps total d’exécution du traitement
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- Date et heure de l’appel
);

-- Table des feedbacks utilisateur
CREATE TABLE chatbot_feedback (
    id SERIAL PRIMARY KEY,
    query_id UUID REFERENCES chatbot_history(query_id) ON DELETE CASCADE,
    username TEXT REFERENCES users(username),
    feedback TEXT CHECK (feedback IN ('Utile', 'Inutile')) NOT NULL,
    comment TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des quiz générés
CREATE TABLE quiz_questions (
    quiz_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username TEXT REFERENCES users(username),
    question TEXT NOT NULL, -- Question générée
    correct_answer TEXT NOT NULL, -- Bonne réponse
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des réponses aux quiz
CREATE TABLE quiz_responses (
    response_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    quiz_id UUID REFERENCES quiz_questions(quiz_id) ON DELETE CASCADE,
    username TEXT REFERENCES users(username),
    user_answer TEXT NOT NULL, -- Réponse donnée par l'utilisateur
    is_correct BOOLEAN NOT NULL, -- Indique si la réponse est correcte
    answered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


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