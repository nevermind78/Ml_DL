# Fiche Matière « Machine Learning et Deep Learning »
**(AU 2025-2026)**

---

## Informations Générales

| Intitulé | Machine Learning et Deep Learning |
|----------|-----------------------------------|
| **Département** | Génie Informatique |
| **Niveau** | 2ème année |
| **Semestre** | S2 |
| **Coefficient** | À définir |
| **Régime** | Mixte |
| **Volume Horaire Total** | 42h |
| **Cours / TD** | 21h |
| **TP / Projet** | 21h |

---

## Prérequis

### Mathématiques
- **Algèbre linéaire** : connaissances sur les vecteurs, les matrices, les opérations matricielles.
- **Calcul différentiel et intégral** : compréhension des dérivées, des gradients et des intégrales.
- **Probabilités et statistiques** : notions de base sur les distributions de probabilité, les estimations de paramètres, les tests d'hypothèses.

### Programmation
- **Compétences de programmation en Python** : compréhension des structures de contrôle, des fonctions, des classes et des bibliothèques couramment utilisées telles que NumPy, Pandas, et Matplotlib.

---

## Objectifs d'Apprentissage

| ID | Objectif |
|----|----------|
| **Obj1** | Comprendre les concepts fondamentaux du Machine Learning. |
| **Obj2** | Être capable d'identifier différents types de problèmes pouvant être résolus avec le ML. |
| **Obj3** | Développer une compréhension des applications du ML dans divers domaines. |
| **Obj4** | Être en mesure d'appliquer des concepts de probabilités et de statistiques pour évaluer les modèles de ML. |
| **Obj5** | Comprendre les différences entre les tâches de régression et de classification. |
| **Obj6** | Être capable d'entraîner et d'évaluer des modèles de régression et de classification. |
| **Obj7** | Explorer les techniques d'optimisation et leurs effets sur la convergence des modèles. |
| **Obj8** | Comprendre les principes du clustering et de la réduction de la dimension. |
| **Obj9** | Être capable d'appliquer des techniques de clustering pour regrouper des données non étiquetées. |
| **Obj10** | Explorer les avantages et les limitations des méthodes de réduction de la dimension. |
| **Obj11** | Acquérir une compréhension approfondie de l'architecture des réseaux de neurones artificiels. |
| **Obj12** | Être capable de construire, entraîner et évaluer des réseaux de neurones multicouches pour des tâches simples. |
| **Obj13** | Comprendre les principes de base de la rétropropagation du gradient. |
| **Obj14** | Comprendre la structure et le fonctionnement des CNN. |
| **Obj15** | Être capable d'appliquer les CNN à des tâches de vision par ordinateur et de traitement d'images. |
| **Obj16** | Explorer des architectures avancées de CNN pour améliorer les performances. |
| **Obj17** | Comprendre et appliquer des méthodes de régularisation avancées (Dropout, BatchNorm). |
| **Obj18** | Savoir utiliser des architectures pré-entraînées (Transfer Learning). |
| **Obj19** | Comprendre les principes des GAN et leurs applications. |
| **Obj20** | Maîtriser les techniques de Gradient Boosting (XGBoost, LightGBM). |

---

## Ouvrages de Référence

| Référence | Détails |
|-----------|---------|
| **Zhou, Victor (2019)** | *"Machine Learning for Beginners: An Introduction to Neural Networks"*, Medium. |
| **Deng & Yu (2014)** | *"Deep Learning: Methods and Applications"*, Foundations and Trends® in Signal Processing. |
| **Goodfellow, Bengio & Courville (2016)** | *"Deep Learning"*, MIT Press. ISBN 978-0-26203561-3. |
| **Géron, A. (2019)** | *"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"*, 2nd Edition. |
| **Chollet, F. (2021)** | *"Deep Learning with Python"*, 2nd Edition, Manning Publications. |

---

## Programme Détaillé des Séances

### **Partie 1 : Machine Learning Fondamental**

| Séance | Type | Contenu | Objectifs Spécifiques |
|--------|------|---------|----------------------|
| **Séance 1** | Cours | **Introduction IA et Machine Learning**<br>- Définitions et concepts de base<br>- Applications et cas d'utilisation<br>- Types d'apprentissage : supervisé, non supervisé, semi-supervisé, renforcement<br>- Étapes de conception d'un modèle IA | Obj1, Obj2, Obj3 |
| **Séance 2** | Cours | **Apprentissage Supervisé : Classification**<br>- Définition et principe<br>- Types : binaire, multi-classes, multi-label<br>- Algorithmes : Arbre de décision, Random Forest, SVM, Naïve Bayes, **Régression Logistique**<br>- Critères d'évaluation | Obj5, Obj6 |
| **Séance 3** | TP | **TP1 : Pipeline de Classification Binaire avec Scikit-learn**<br>- Chargement et préparation des données<br>- Split train/validation/test<br>- Standardisation dans un pipeline<br>- Entraînement et évaluation initiale | Obj6, Obj7 |
| **Séance 4** | TD | **TD1 : Modèles de Classification de Base**<br>- Arbre de décision<br>- Régression logistique (intuition probabiliste)<br>- k-NN : distance et voisinage<br>- Naïve Bayes<br>- **Gradient Boosting (XGBoost/LightGBM)**<br>- Avantages, limites, complexité | Obj4, Obj6 |
| **Séance 5** | TD | **TD2 : Critères d'Évaluation**<br>- Matrice de confusion<br>- Accuracy, Precision, Recall, F1-score<br>- Courbe ROC/AUC<br>- Comparaison critique des résultats | Obj4 |
| **Séance 6** | TP | **TP2 : Classification Multi-classes & Optimisation**<br>- Utilisation de GridSearchCV et RandomizedSearchCV<br>- Validation croisée<br>- Comparaison de plusieurs algorithmes | Obj6, Obj7 |
| **Séance 7** | Cours | **Apprentissage Supervisé : Régression**<br>- Définition et principe<br>- Régression linéaire : fondements théoriques<br>- **Régularisation : Ridge, Lasso, ElasticNet**<br>- SVR (Support Vector Regression)<br>- Métriques d'évaluation (MAE, MSE, R²) | Obj5, Obj6 |
| **Séance 8** | TP | **TP3 : Régression & Optimisation**<br>- Mise en œuvre de modèles de régression<br>- Optimisation des hyperparamètres<br>- Évaluation et comparaison | Obj6, Obj7 |
| **Séance 9** | Cours | **Apprentissage Non Supervisé**<br>- Définitions et principes<br>- k-means : avantages, inconvénients<br>- DBSCAN : principe de densité<br>- Mesures de qualité (silhouette, inertie)<br>- Applications réelles | Obj8, Obj9 |
| **Séance 10** | TP | **TP4 : Clustering & Réduction de Dimension**<br>- Choix du nombre de clusters<br>- Visualisation des groupes<br>- PCA pour amélioration de l'analyse<br>- Interprétation métier | Obj9, Obj10 |

### **Partie 2 : Réseaux de Neurones & Deep Learning**

| Séance | Type | Contenu | Objectifs Spécifiques |
|--------|------|---------|----------------------|
| **Séance 11** | Cours | **Réseaux de Neurones Artificiels**<br>- Architecture (couches, neurones)<br>- Fonctions d'activation<br>- Fonction coût<br>- Descente de gradient<br>- Rétropropagation<br>- Paramètres & Hyperparamètres<br>- Régularisation (Dropout, BatchNorm) | Obj11, Obj12, Obj13, Obj17 |
| **Séance 12** | TP | **TP5 : Réseaux de Neurones avec Keras/TensorFlow**<br>- Construction d'un réseau multicouche<br>- Choix des fonctions d'activation<br>- Entraînement avec SGD/Adam<br>- Visualisation des courbes loss/accuracy<br>- Techniques de régularisation | Obj12, Obj13, Obj17 |
| **Séance 13** | TD | **TD de Révision : QCM & Préparation**<br>- Révision des concepts ML/RNA<br>- Exercices pratiques<br>- Préparation aux certifications | Tous objectifs |
| **Séance 14** | DS | **Devoir Surveillé**<br>- Partie théorique (QCM, questions ouvertes)<br>- Partie pratique (analyse de code, interprétation de résultats) | Évaluation intermédiaire |
| **Séance 15** | Cours | **Deep Learning : Fondements Avancés**<br>- Différence RNA/DL<br>- Architectures : CNN, RNN, LSTM<br>- Principes des CNN (convolution, pooling)<br>- Applications en vision par ordinateur | Obj14, Obj15, Obj16 |
| **Séance 16** | TP | **TP6 : Réseaux Convolutifs (CNN)**<br>- Construction d'un CNN simple<br>- Entraînement sur dataset d'images<br>- Visualisation des filtres<br>- Évaluation des performances | Obj15, Obj16 |
| **Séance 17** | TP | **TP7 : Transfer Learning avec CNN**<br>- Utilisation de modèles pré-entraînés (VGG16, ResNet)<br>- Fine-tuning<br>- Évaluation comparative | Obj15, Obj18 |
| **Séance 18** | TP | **TP8 : RNN pour Séries Temporelles**<br>- Prétraitement des séquences<br>- Construction de LSTM/GRU<br>- Prédiction de séries temporelles<br>- Évaluation et visualisation | Obj12, Obj13 |
| **Séance 19** | Cours | **IA Générative & GAN**<br>- Définition et principes<br>- Architecture GAN (Générateur/Discriminateur)<br>- Fonctions de coût (adversarial loss)<br>- Applications en Data Science | Obj19 |
| **Séance 20** | TP | **TP9 : Introduction aux GAN**<br>- Initialisation des modèles<br>- Définition des loss et optimizers<br>- Entraînement itératif<br>- Visualisation des images générées | Obj19 |

### **Partie 3 : Projet & Évaluation Finale**

| Séance | Type | Contenu | Objectifs Spécifiques |
|--------|------|---------|----------------------|
| **Séance 21** | Cours | **Proposition de Mini-Projet**<br>- Présentation des règles<br>- Proposition des sujets par étudiants<br>- Validation des datasets et faisabilité<br>- Définition des livrables | Application des concepts |
| **Séance 22** | Projet | **Travail Encadré sur Projet**<br>- Séance de travail en groupe<br>- Assistance technique et méthodologique | Obj3, Obj6, Obj12, Obj15 |
| **Séance 23** | Projet | **Travail Encadré sur Projet**<br>- Finalisation du code<br>- Préparation du rapport et de la présentation | Application complète |
| **Séance 24** | Évaluation | **Présentation des Mini-Projets**<br>- Présentation orale (10 min/groupe)<br>- Démonstration technique<br>- Questions/Réponses | Évaluation finale |

---

## Système d'Évaluation

| Type d'Évaluation | Pourcentage | Détails |
|-------------------|-------------|---------|
| **Mini-Projet** | 60% | - Rapport technique (20%)<br>- Code et implémentation (20%)<br>- Présentation orale (20%) |
| **Devoir Surveillé (DS)** | 40% | - Partie théorique : QCM/questions ouvertes (20%)<br>- Partie pratique : analyse de code/cas d'étude (20%) |

---

## Responsables

| Rôle | Nom & Prénom |
|------|--------------|
| **Enseignant(e)** | Dr. Fatma Sbiaa |
| **Directeur du Département** | Mr. Ramzi Mahmoudi |
| **Directeur des Études** | Pr. Moncef Bouzidi |

---

## Notes Pédagogiques

1. **Supports de cours** : Disponibles sur plateforme Moodle/Teams
2. **Langages/Frameworks** : Python, Scikit-learn, TensorFlow/Keras, PyTorch (optionnel)
3. **Environnement** : Jupyter Notebook, Google Colab, VS Code
4. **Projets** : Les étudiants peuvent proposer leurs propres datasets (sous validation)
5. **Certifications** : Préparation aux certifications Google/Microsoft en ML/Datacamp

