# RePilot Recycle Chatbot 🤖♻️

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0-blue)](https://www.postgresql.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0-green)](https://www.mongodb.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.15-orange)](https://www.trychroma.com/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-1.0-yellow)](https://github.com/BerriAI/litellm)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.2.2-blueviolet)](https://www.sbert.net/)
[![EcoLogits](https://img.shields.io/badge/EcoLogits-Enabled-green)](https://github.com/your-org/ecologits)
[![Mistral](https://img.shields.io/badge/Mistral%20AI-API-purple)](https://mistral.ai/)

## Table des matières

- [Présentation](#présentation)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du projet](#architecture-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Contribution](#contribution)
- [Auteurs](#auteurs)

## Présentation

RePilot est un chatbot intelligent spécialisé dans le tri des déchets. Utilisant le RAG (Retrieval-Augmented Generation) et l'API Mistral, il fournit des réponses contextualisées selon la ville de l'utilisateur tout en mesurant son impact environnemental (choix de la taille du modèle).

## Fonctionnalités

- 🎯 **RAG Intelligent**
  - Reformulation automatique des questions
  - Réponses basées sur le contexte local (Système de tri des déchets selon la ville Paris/Grand Métropoel de Lyon)
  - Système de garde-fou pour requêtes inappropriées

- 📊 **Monitoring Complet**
  - Suivi de la latence des requêtes
  - Mesure de l'impact environnemental (GWP)
  - Analyse de la consommation d'énergie
  - Calcul des coûts par requête

  - 🗄️ **Choix du Data Warehouse**
  - **PostgreSQL** : Base de données relationnelle puissante pour le scaling
  - **MongoDB** : Base NoSQL adaptée aux instructions de recyclage (clé:valeur) pour un apprentissage efficace du RAG
  - **ChromaDB** : Stockage des vecteurs (embeddings) en chunks avec SentenceTransformer

- 👥 **Gestion Utilisateurs**
  - Interface admin dédiée
  - Système de quiz interactif
  - Historique des conversations
  - Feedback utilisateur

## Architecture du projet

```
RePilot-recycle-chatbot/
├── assets/
├── chromadb/
├── data/
│   ├── grand_lyon_processed.json
│   ├── Paris_tri.json
├── database/
│   ├── __init__.py
│   ├── db_management.py
├── guardrail/
│   ├── storage/
│   │   ├── guardrail.pkl
│   ├── service.py
├── plots/
│   ├── clusters.py
│   ├── plots.py
├── rag_simulation/
│   ├── corpus_ingestion.py
│   ├── embeddings.py
│   ├── rag_augmented.py
│   ├── schema.py
│   ├── wrapper.py
├── sql/
│   ├── init.sql
├── views/
│   ├── admin_dashboard.py
│   ├── lm.py
│   ├── login.py
│   ├── user_dashboard.py
├── .dockerignore
├── .env
├── .gitignore
├── app.py
├── config.yml
├── docker-compose.yml
├── dockerfile
├── LICENSE
├── notebook_training_gr.ipynb
├── notebook_training_gr.py
├── README.md
├── requirements.txt
├── temp.ipynb
├── temp.txt
├── users.json
├── utils.py
```

## Installation

### Prérequis

- Python 3.9+
- PostgreSQL
- Compte Mistral AI
- Un fichier `.env` contenant vos clés API (par exemple, `MISTRAL_API_KEY`)

### Étapes d'installation

```bash
# Cloner le dépôt
git clone https://github.com/jdalfons/RePilot-recycle-chatbot.git
cd RePilot-recycle-chatbot

# Créer et activer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux

# Installer les dépendances
pip install -r requirements.txt
```

### Configuration de la base de données

```bash
psql -U postgres -f database/init.sql
```

### Configuration des variables d'environnement

Créer un fichier `.env` et y ajouter les informations suivantes :

```ini
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=poc_rag
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
MISTRAL_API_KEY=your_api_key
```

## Utilisation

```bash
streamlit run app.py
```

### Accès

- **Interface utilisateur**  :
  - **Login** : `user`
  - **Password** : `user123`
- **Dashboard admin** :
  - **Login** : `admin`
  - **Password** : `admin123`

## Contribution

1. Forker le projet
2. Créer une branche :

   ```bash
   git checkout -b feature/NouvelleFonctionnalite
   ```

3. Commiter vos modifications :

   ```bash
   git commit -m "Ajout nouvelle fonctionnalité"
   ```

4. Pousser vers le dépôt distant :

   ```bash
   git push origin feature/NouvelleFonctionnalite
   ```

5. Créer une Pull Request

## Auteurs

### Core Team

- [Juan Diego A.](https://github.com/jdalfons) 
- [Quentin Lim](https://github.com/QL2111) 
- [ADJARO](https://github.com/Adjaro) 
- [Akrem Jomaa](https://github.com/akremjomaa) 
- [Yacine Ayachi]()

### Contributions

[![Contributors](https://contrib.rocks/image?repo=jdalfons/RePilot-recycle-chatbot)](https://github.com/jdalfons/RePilot-recycle-chatbot/graphs/contributors)

_Merci à tous les contributeurs qui ont participé à ce projet !_
