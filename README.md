# RePilot Recycle Chatbot

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![EcoLogits](https://img.shields.io/badge/EcoLogits-Enabled-green)](https://github.com/your-org/ecologits)

**RePilot Recycle Chatbot** est un assistant virtuel intelligent conçu pour aider les utilisateurs à trier leurs déchets de manière efficace et écologique. Ce chatbot utilise des techniques avancées de traitement du langage naturel (NLP) et de recherche augmentée (RAG) pour fournir des réponses précises et contextualisées sur le recyclage.

## Fonctionnalités

- **Reformulation des requêtes** : Corrige les fautes d'orthographe et de syntaxe dans les questions des utilisateurs.
- **Contexte localisé** : Fournit des informations spécifiques à la ville de l'utilisateur (par exemple, Paris ou Grand Lyon Métropole).
- **Monitoring des performances** : Mesure la latence, le coût, l'énergie utilisée et l'impact environnemental (GWP) des appels au modèle de langage.
- **Sécurité des requêtes** : Utilise un système de garde-fou (guardrail) pour détecter et bloquer les requêtes inappropriées.
- **Base de données intégrée** : Stocke les requêtes et les réponses pour une analyse ultérieure.

## Installation

### Prérequis

- Python 3.9 ou supérieur
- Un fichier `.env` contenant vos clés API (par exemple, `MISTRAL_API_KEY`)

### Étapes d'installation

1. Clonez le dépôt :

   ```bash
   git clone https://github.com/jdalfons/RePilot-recycle-chatbot.git
   cd RePilot-recycle-chatbot
