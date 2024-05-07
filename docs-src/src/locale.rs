use leptos::*;
use leptos_router::*;

#[derive(Clone, Default, Eq, PartialEq)]
pub enum Languages {
    #[default]
    English,
    French,
}

#[derive(Params, PartialEq)]
pub struct SiteQueries {
    pub lang: String,
}

impl Default for SiteQueries {
    fn default() -> Self {
        Self {
            lang: "en".to_owned(),
        }
    }
}

#[derive(Clone, Default)]
pub struct Locale {
    pub lang: Languages,
}

impl Locale {
    pub fn new(lang: Languages) -> Self {
        Self { lang }
    }
    pub fn toggle_lang(&mut self) {
        if self.lang == Languages::English {
            self.lang = Languages::French;
        } else {
            self.lang = Languages::English;
        }
    }

    pub fn get_progress_title(&self) -> String {
        if self.lang == Languages::English {
            "Progress".to_owned()
        } else {
            "Progrès".to_owned()
        }
    }

    pub fn get_lang_button_text(&self) -> String {
        if self.lang == Languages::English {
            "Toggle Language".to_owned()
        } else {
            "Changer de Langue".to_owned()
        }
    }

    pub fn get_sidebar_title(&self) -> String {
        if self.lang == Languages::English {
            "IFT4055 - Honors Project".to_owned()
        } else {
            "IFT4055 - Projet Honor".to_owned()
        }
    }
    pub fn get_sidebar_description(&self) -> String {
        if self.lang == Languages::English {
            "Built with love using Rust 🦀 and NeoVim 🖥️".to_owned()
        } else {
            "Construit avec passion en utilisant Rust 🦀 et NeoVim 🖥️".to_owned()
        }
    }
    pub fn get_planning(&self) -> String {
        let en = "
            <p class=\"text-lighttext-800 dark:text-darktext-200\">
                This entire project should span the 4 months of summer 2024. Here is a general outlook on the schedule.
            </p>
            <ul class=\"list-disc list-outside pl-8 lg:pl-6\">
                <li>May: Planning, reading and initial research</li>
                <li>June: Implementation of models</li>
                <li>July: Debugging and generation of results</li>
                <li>August: Writing for paper</li>
            </ul>
        ".to_owned();
        let fr = "
            <p class=\"text-lighttext-800 dark:text-darktext-200\">
                L'ensemble de ce projet devrait s'étendre sur les 4 mois de l'été 2024. Voici un aperçu général du calendrier.
            </p>
            <ul class=\"list-disc list-outside pl-8 lg:pl-6\">
                <li>May: Planification, lecture et recherche initiale</li>
                <li>June: Implémentation des modèles</li>
                <li>July: Débogage et génération des résultats</li>
                <li>August: Rédaction du document</li>
            </ul>
        ".to_owned();
        if self.lang == Languages::English {
            en
        } else {
            fr
        }
    }
    pub fn get_description_title(&self) -> String {
        if self.lang == Languages::English {
            "Efficient Function Evaluation Using Reinforcement Learning".to_owned()
        } else {
            "Évaluation efficace des fonctions à l'aide de l'apprentissage par renforcement"
                .to_owned()
        }
    }
    pub fn get_description(&self) -> String {
        let en = "
                    <p>
                        In scientific endeavors, the evaluation of complex functions, often spanning multiple dimensions, presents a significant computational challenge. These functions can be costly and time-consuming to evaluate accurately. The project proposes the development of a novel approach to address this challenge by leveraging neural networks and reinforcement learning techniques to emulate the evaluation of arbitrarily complex functions efficiently.
                    </p>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Objectives</h3>
                    <p>
                        The primary objective of the research project is to train a neural network capable of emulating the evaluation of complex functions in a fraction of the time it would take to evaluate the true function directly. Specifically, the project aims to:
                    </p>
                    <ul class=\"list-disc list-outside pl-8 lg:pl-6\">
                        <li>
                            Develop a neural network architecture capable of accurately emulating complex functions.
                        </li>
                        <li>
                            Train the neural network efficiently using reinforcement learning techniques to sample points in the function space effectively.
                        </li>
                        <li>
                            Minimize the computational cost associated with evaluating complex functions by optimizing the sampling strategy.
                        </li>
                    </ul>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Methodology</h3>
                    <p>The research will employ a two-step approach:</p>
                    <ol class=\"list-decimal list-outside pl-8 lg:pl-6\">
                        <li>
                            <strong>Reinforcement Learning for Efficient Sampling: </strong>
                            A reinforcement learning model will be trained to sample points in the function space effectively. The model's objective will be to learn how to select points that contribute the most to accurately representing the function while minimizing the number of evaluations required. The reinforcement learning model will use a non-differentiable function as a reward signal, representing the discrepancy between the true function and its emulation. This function will guide the model to sample points efficiently.
                        </li>
                        <li>
                            <strong>Neural Network Emulation: </strong>
                            A neural network will be trained using the sampled points generated by the reinforcement learning model. This neural network will learn to approximate the complex function using the sampled data points. By emulating the function, the neural network will enable rapid evaluation of the function at any given input, significantly reducing computational overhead compared to direct evaluation.
                        </li>
                    </ol>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Expected Outcomes</h3>
                    <ul class=\"list-disc list-outside pl-8 lg:pl-6\">
                        <li>
                            Development of a novel approach for efficiently evaluating complex functions using neural networks and reinforcement learning.
                        </li>
                        <li>
                            A trained neural network capable of accurately emulating complex functions, significantly reducing evaluation time.
                        </li>
                        <li>
                            Insights into efficient sampling strategies for complex function spaces, applicable beyond the scope of this research.
                        </li>
                    </ul>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">
                        Significance and Potential Applications
                    </h3>
                    <p>
                        Efficient function evaluation is critical in various scientific disciplines, including physics, engineering, and machine learning. The proposed approach has the potential to revolutionize computational methods by enabling faster and more efficient evaluation of complex functions. Applications include optimization problems, simulation-based analysis, and model training in various domains.
                    </p>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Timeline and Resources</h3>
                    <p>
                        The research will be conducted over a specified timeline, leveraging computational resources for model training and validation. Collaboration with experts in machine learning, optimization, and domain-specific areas will enrich the research process and ensure its success.
                    </p>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Conclusion</h3>
                    <p>
                        The project proposes a novel approach to address the computational challenges associated with evaluating complex functions efficiently. By combining neural networks and reinforcement learning techniques, the research aims to develop a scalable solution applicable to a wide range of scientific and engineering domains. This innovative approach has the potential to accelerate progress in computational science and enable breakthroughs in complex problem-solving.
                    </p>".to_owned();
        let fr = "

                <div class=\"text-lighttext-800 dark:text-darktext-200\">
                    <p>
                        En sciences, l'évaluation de fonctions complexes, souvent s'étendant sur plusieurs dimensions, présente un défi computationnel significatif. Ces fonctions peuvent être coûteuses et prendre beaucoup de temps à évaluer avec précision. Le projet propose le développement d'une approche novatrice pour relever ce défi en tirant parti des réseaux neuronaux et des techniques d'apprentissage par renforcement pour émuler efficacement l'évaluation de fonctions arbitrairement complexes.
                    </p>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Objectifs</h3>
                    <p>
                        L'objectif principal du projet de recherche est de former un réseau neuronal capable d'émuler l'évaluation de fonctions complexes en une fraction du temps nécessaire pour évaluer directement la vraie fonction. Plus précisément, le projet vise à :
                    </p>
                    <ul class=\"list-disc list-outside pl-8 lg:pl-6\">
                        <li>
                            Développer une architecture de réseau neuronal capable d'émuler avec précision des fonctions complexes.
                        </li>
                        <li>
                            Entraîner le réseau neuronal de manière efficace en utilisant des techniques d'apprentissage par renforcement pour échantillonner efficacement des points dans l'espace des fonctions.
                        </li>
                        <li>
                            Minimiser le coût computationnel associé à l'évaluation de fonctions complexes en optimisant la stratégie d'échantillonnage.
                        </li>
                    </ul>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Méthodologie</h3>
                    <p>La recherche utilisera une approche en deux étapes :</p>
                    <ol class=\"list-decimal list-outside pl-8 lg:pl-6\">
                        <li>
                            <strong>Apprentissage par Renforcement pour un Échantillonnage Efficace : </strong>
                            Un modèle d'apprentissage par renforcement sera formé pour échantillonner efficacement des points dans l'espace des fonctions. L'objectif du modèle sera d'apprendre à sélectionner des points qui contribuent le plus à représenter avec précision la fonction tout en minimisant le nombre d'évaluations nécessaires. Le modèle d'apprentissage par renforcement utilisera une fonction non différentiable comme signal de récompense, représentant la disparité entre la vraie fonction et son émulation. Cette fonction guidera le modèle pour échantillonner des points efficacement.
                        </li>
                        <li>
                            <strong>Émulation par Réseau Neuronal : </strong>
                            Un réseau neuronal sera formé en utilisant les points échantillonnés générés par le modèle d'apprentissage par renforcement. Ce réseau neuronal apprendra à approximer la fonction complexe en utilisant les points de données échantillonnés. En émulant la fonction, le réseau neuronal permettra une évaluation rapide de la fonction à n'importe quelle entrée donnée, réduisant ainsi considérablement les frais informatiques par rapport à une évaluation directe.
                        </li>
                    </ol>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Résultats Attendus</h3>
                    <ul class=\"list-disc list-outside pl-8 lg:pl-6\">
                        <li>
                            Développement d'une approche novatrice pour évaluer efficacement des fonctions complexes en utilisant des réseaux neuronaux et l'apprentissage par renforcement.
                        </li>
                        <li>
                            Un réseau neuronal formé capable d'émuler avec précision des fonctions complexes, réduisant significativement le temps d'évaluation.
                        </li>
                        <li>
                            Perspectives sur les stratégies d'échantillonnage efficaces pour les espaces de fonctions complexes, applicables au-delà du champ de cette recherche.
                        </li>
                    </ul>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">
                        Signification et Applications Potentielles
                    </h3>
                    <p>
                        L'évaluation efficace des fonctions est cruciale dans diverses disciplines scientifiques, notamment la physique, l'ingénierie et l'apprentissage automatique. L'approche proposée a le potentiel de révolutionner les méthodes computationnelles en permettant une évaluation plus rapide et plus efficace de fonctions complexes. Les applications incluent les problèmes d'optimisation, l'analyse basée sur la simulation et la formation de modèles dans divers domaines.
                    </p>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Calendrier et Ressources</h3>
                    <p>
                        La recherche sera menée sur une période définie, en utilisant des ressources informatiques pour la formation et la validation du modèle. La collaboration avec des experts en apprentissage automatique, en optimisation et dans des domaines spécifiques enrichira le processus de recherche et garantira son succès.
                    </p>

                    <br/>
                    <h3 class=\"text-lg text-lightaccent-600 dark:text-darkaccent-400\">Conclusion</h3>
                    <p>
                        Le projet propose une approche novatrice pour relever les défis computationnels associés à l'évaluation efficace de fonctions complexes. En combinant des réseaux neuronaux et des techniques d'apprentissage par renforcement, la recherche vise à développer une solution évolutive applicable à un large éventail de domaines scientifiques et d'ingénierie. Cette approche innovante a le potentiel d'accélérer les progrès en sciences computationnelles et de permettre des percées dans la résolution de problèmes complexes.
                    </p>
                </div>".to_owned();

        if self.lang == Languages::English {
            en
        } else {
            fr
        }
    }
}
