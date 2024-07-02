#[derive(Clone, Default, Eq, PartialEq)]
pub enum Languages {
    #[default]
    English,
    French,
}

#[derive(Clone, Default)]
pub struct Locale<'a> {
    pub progress_1_title: &'a str,
    pub progress_1_description: &'a str,
    pub progress_2_title: &'a str,
    pub progress_2_description: &'a str,
    pub progress_3_title: &'a str,
    pub progress_3_description: &'a str,
    pub progress_4_title: &'a str,
    pub progress_4_description: &'a str,
    pub progress_title: &'a str,
    pub lang_button_text: &'a str,
    pub sidebar_title: &'a str,
    pub sidebar_description: &'a str,
    pub planning_title: &'a str,
    pub planning: &'a str,
    pub description_title: &'a str,
    pub description: &'a str,
}

pub const LOCALE_EN: Locale<'static> = Locale {
    progress_1_title: "Creation of website, getting familiar with problem and reading",
    progress_1_description: "
        To optimize the sampling process in the reinforcement learning network, the reward function must to quantify how well the points provided by the network represent the function from which they are sampled. Put simply, it should measure the disparity between the original function and the sampled points generated by the model.

        <br/>
        <a href=\"https://arxiv.org/abs/2402.04355\">PQMass</a> serves as a viable candidate for this purpose, as it quantifies the probability that two sets of samples originate from the same distribution. Therefore, by uniformly sampling points from simple functions and evaluating them against the model's output using the aforementioned metric, the model can learn to sample efficiently, ensuring that the points accurately represent the function with minimal sampling.

        <br/>
        A question remains: how to model the state/the actions of the model and the continuous action space that shouldn't be discretized? How should the model update how it samples points given its reward/score?
    ",
    progress_2_title: "Continuous RL, implementation, RL workshop and move to GFlowNets",
    progress_2_description: "
        To solve the issue of discretization, the actor-critic algorithm was researched, allowing for continuous action spaces by creating a model which returns the probability distribution of the actions from which it is possible to sample an action. A basic implementation of the actor-critic algorith was created with PQMass as the reward function. I also attended a MILA workshop on RL 

        <br/>
        After discussing with my supervisor, it was decided that classic RL (DQN, actor-critic, etc) was not the best approach for my problem. Indeed, these methods have a harder time learning the various modes that can be present in the function space. GFlowNets offer an architecture which explores that function space better. Research was done to understand how GFlowNets work and how they can be used to sample points in a function space. A simple implementation of GFlowNets, sampling basic functions, was created following tutorials from the <a href=\"https://github.com/GFNOrg/torchgfn/\">torchgfn python package</a>. GFlowNets seem promising!
    ",
    progress_3_title: "GFlowNets, Bayesian Optimisation, Gaussian Processes and Acquisition Function",
    progress_3_description: "
        After working on the GFlowNet implementation, it is clear that this architecture is promising for the problem at hand. That being said, the GFlowNet used had trouble finding all of the modes in a sparse distribution. This is probably caused by the usage of an \"on-policy\" model, meaning that the model fould sample its next points from the learned policy. This should not be the case in the final project, because the acquisition model will be used to propose the points.

        <br/>
        Up until now, it is therefore clear that a GFlowNet will be used as a surrogate model to emulate/approximate the complex distribution/function. Moreover after discussing with my supervisors, I will be exploring how gaussian processes and acquisition functions (AF) are used in the context of bayesian optimisation (BO). At first glance, it seems that this project is closely related to BO. In fact, the main difference is that BO only cares about accurately finding the maximal/minimal value of a function (because it is an optimisation problem), whereas this project aims to accurately model the entire function and sample from it.

        <br/>
        The next step is to research on papers discussing the use of neural networks as AFs in BO in order to learn how they work and how to adapt them to my problem that is not an optimisation problem.
    ",
    progress_4_title: "Few Shot Learning and Meta Learning",
    progress_4_description: "
        After reading on BO, I stumbled on papers researching the use of meta learning and few shot learning in the context of BO. This is interesting because it could potentially help in the context of my project as my acquisition network should be able to learn from a class of problems and be able to make decisions on new problems it has never seen before (assuming it is also in that class of problems).

        <br/>
        Reading these papers and their code was interesting, but I was lacking the theoretical background to understand everything. I therefore followed online courses on meta learning and few shot learning before revisiting these papers.

        <br/>
        I also met a PhD student with my supervisor at MILA who used to work on BO. This meeting was helpful and it was concluded that I should try to find a way to introduce some notion of uncertainty/variance in the surrogate network. That is, instead of outputting a point y for a given x, the surrogate network should output the mean and variance of y for a given x. This will be used to inform the acquisition model where it should sample next.
    ",
    ",
    progress_title: "Progress",
    lang_button_text: "Toggle Language",
    sidebar_title: "IFT4055 - Honors Project",
    sidebar_description: "Built with love using Rust 🦀 and NeoVim 🖥️",
    planning_title: "Project Schedule",
    planning: "
        <p class=\"text-lighttext-800 dark:text-darktext-200\">
            This entire project should span the 4 months of summer 2024. Here is a general outlook on the schedule.
        </p>
        <ul>
            <li>May: Planning, reading and initial research</li>
            <li>June: Implementation of models</li>
            <li>July: Debugging and generation of results</li>
            <li>August: Writing for paper</li>
        </ul>",
    description_title: "Efficient Function Evaluation Using Reinforcement Learning",
    description: "
        <p>
            In scientific endeavors, the evaluation of complex functions, often spanning multiple dimensions, presents a significant computational challenge. These functions can be costly and time-consuming to evaluate accurately. The project proposes the development of a novel approach to address this challenge by leveraging neural networks and reinforcement learning techniques to emulate the evaluation of arbitrarily complex functions efficiently.
        </p>

        <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Objectives</h3>
        <p>
            The primary objective of the research project is to train a neural network capable of emulating the evaluation of complex functions in a fraction of the time it would take to evaluate the true function directly. Specifically, the project aims to:
        </p>
        <ul>
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

        <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Methodology</h3>
        <p>The research will employ a two-step approach:</p>
        <ol>
            <li>
                <strong>Reinforcement Learning for Efficient Sampling: </strong>
                A reinforcement learning model will be trained to sample points in the function space effectively. The model's objective will be to learn how to select points that contribute the most to accurately representing the function while minimizing the number of evaluations required. The reinforcement learning model will use a non-differentiable function as a reward signal, representing the discrepancy between the true function and its emulation. This function will guide the model to sample points efficiently.
            </li>
            <li>
                <strong>Neural Network Emulation: </strong>
                A neural network will be trained using the sampled points generated by the reinforcement learning model. This neural network will learn to approximate the complex function using the sampled data points. By emulating the function, the neural network will enable rapid evaluation of the function at any given input, significantly reducing computational overhead compared to direct evaluation.
            </li>
        </ol>

        <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Expected Outcomes</h3>
        <ul>
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

        <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">
            Significance and Potential Applications
        </h3>
        <p>
            Efficient function evaluation is critical in various scientific disciplines, including physics, engineering, and machine learning. The proposed approach has the potential to revolutionize computational methods by enabling faster and more efficient evaluation of complex functions. Applications include optimization problems, simulation-based analysis, and model training in various domains.
        </p>

        <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Timeline and Resources</h3>
        <p>
            The research will be conducted over a specified timeline, leveraging computational resources for model training and validation. Collaboration with experts in machine learning, optimization, and domain-specific areas will enrich the research process and ensure its success.
        </p>

        <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Conclusion</h3>
        <p>
            The project proposes a novel approach to address the computational challenges associated with evaluating complex functions efficiently. By combining neural networks and reinforcement learning techniques, the research aims to develop a scalable solution applicable to a wide range of scientific and engineering domains. This innovative approach has the potential to accelerate progress in computational science and enable breakthroughs in complex problem-solving.
        </p>
    ",
};

pub const LOCALE_FR: Locale<'static> = Locale {
    progress_1_title: "Création du site web, familiarisation avec le problème et lecture",
    progress_1_description: "
        Pour optimiser le processus d'échantillonnage dans le réseau d'apprentissage par renforcement, la fonction de récompense doit quantifier la qualité des points fournis en s'assurant qu'ils représentent la fonction à partir de laquelle ils sont échantillonnés. En d'autres termes, elle doit mesurer la disparité entre la fonction originale et les points échantillonnés générés par le modèle.

        <br/>
        <a href=\"https://arxiv.org/abs/2402.04355\">PQMass</a> constitue un candidat viable à cette fin, car cette technique quantifie la probabilité que deux ensembles d'échantillons proviennent de la même distribution. Par conséquent, en échantillonnant uniformément des points à partir de fonctions simples et en les évaluant par rapport à la sortie du modèle en utilisant la métrique mentionnée précédemment, le modèle peut apprendre à échantillonner efficacement, garantissant que les points représentent avec précision la fonction avec un échantillonnage minimal.

        <br/>
        Une question demeure: comment modéliser l'état/les actions du modèle et comment gérer l'espace d'actions continu qui ne doit pas être discrétisé? Comment le modèle devrait-il mettre à jour sa manière d'échantillonner des points compte tenu de sa récompense/son score?
    ",
    progress_2_title: "RL continu, implémentation, atelier RL et transition vers les GFlowNets",
    progress_2_description: "
        Pour résoudre le problème de discrétisation, l'algorithme acteur-critique a été étudié, permettant des espaces d'action continus en créant un modèle qui retourne la distribution de probabilité des actions à partir de laquelle il est possible d'échantillonner une action. Une implémentation de base de l'algorithme acteur-critique a été créée avec PQMass comme fonction de récompense. J'ai également assisté à un atelier de MILA sur l'apprentissage par renforcement.

        <br/>
        Après avoir discuté avec mon superviseur, il a été décidé que l'apprentissage par renforcement classique (DQN, acteur-critique, etc.) n'était pas la meilleure approche pour mon problème. En effet, ces méthodes ont plus de difficulté à apprendre les différentes modalités pouvant être présentes dans l'espace fonctionnel. Les GFlowNets offrent une architecture qui explore mieux cet espace fonctionnel. Des recherches ont été effectuées pour comprendre comment fonctionnent les GFlowNets et comment ils peuvent être utilisés pour échantillonner des points dans un espace fonctionnel. Une implémentation simple de GFlowNets, échantillonnant des fonctions de base, a été créée en suivant les tutoriels du <a href=\"https://github.com/GFNOrg/torchgfn/\">package python torchgfn</a>. Les GFlowNets semblent prometteurs!
    ",
    progress_3_title: "GFlowNets, optimisation bayésienne, processus gaussiens et fonctions d'acquisitions",
    progress_3_description: "
        Après avoir travaillé sur l'implémentation de GFlowNet, il est clair que cette architecture est prometteuse pour le problème en question. Cela étant dit, le GFlowNet utilisé a eu des difficultés à trouver tous les modes dans une distribution éparse. Cela est probablement dû à l'utilisation d'un modèle \"on-policy\", ce qui signifie que le modèle échantillonnait ses prochains points à partir de la politique apprise. Cela ne devrait pas être le cas dans le projet final, car le modèle d'acquisition sera utilisé pour proposer les points.

        <br/>
        Jusqu'à présent, il est donc clair qu'un GFlowNet sera utilisé comme modèle de substitution pour émuler/approximer la distribution/fonction complexe. De plus, après avoir discuté avec mes superviseurs, j'explorerai comment les processus gaussiens et les fonctions d'acquisition (FA) sont utilisés dans le contexte de l'optimisation bayésienne (OB). À première vue, il semble que ce projet soit étroitement lié à l'OB. En fait, la principale différence est que l'OB se préoccupe uniquement de trouver avec précision la valeur maximale/minimale d'une fonction (car c'est un problème d'optimisation), alors que ce projet vise à modéliser avec précision toute la fonction et à en échantillonner.

        <br/>
        La prochaine étape consiste à rechercher des articles discutant de l'utilisation des réseaux neuronaux comme FAs dans l'OB afin d'apprendre comment ils fonctionnent et comment les adapter à mon problème qui n'est pas un problème d'optimisation.
    ",
    progress_4_title: "Apprentissage par quelques exemples et méta-apprentissage",
    progress_4_description: "
        Après avoir lu sur l'optimisation bayésienne (OB), je suis tombé sur des articles de recherche concernant l'utilisation du méta-apprentissage et de l'apprentissage par petits échantillons dans le contexte de l'OB. C'est intéressant car cela pourrait potentiellement aider dans le cadre de mon projet, étant donné que mon réseau d'acquisition devrait être capable d'apprendre à partir d'une classe de problèmes et être capable de prendre des décisions sur de nouveaux problèmes qu'il n'a jamais vus auparavant (en supposant qu'ils appartiennent également à cette classe de problèmes).

        <br/>
        La lecture de ces articles et de leur code était intéressante, mais il me manquait le bagage théorique pour tout comprendre. J'ai donc suivi des cours en ligne sur le méta-apprentissage et l'apprentissage par petits échantillons avant de retourner à ces articles.

        <br/>
        J'ai également rencontré un étudiant en doctorat avec mon superviseur à MILA, qui travaillait auparavant sur l'OB. Cette rencontre a été utile et il a été conclu que je devrais essayer de trouver un moyen d'introduire une notion d'incertitude/de variance dans le réseau de substitution. C'est-à-dire que, au lieu de donner un point y pour un x donné, le réseau de substitution devrait donner la moyenne et la variance de y pour un x donné. Cela sera utilisé pour informer le modèle d'acquisition sur l'endroit où il doit échantillonner ensuite.
    ",
    progress_title: "Progrès",
    lang_button_text: "Changer de Langue",
    sidebar_title: "IFT4055 - Projet Honor",
    sidebar_description: "Construit avec passion en utilisant Rust 🦀 et NeoVim 🖥️",
    planning_title: "Calendrier du Projet",
    planning: "
        <p class=\"text-lighttext-800 dark:text-darktext-200\">
            L'ensemble de ce projet devrait s'étendre sur les 4 mois de l'été 2024. Voici un aperçu général du calendrier.
        </p>
        <ul>
            <li>Mai: Planification, lecture et recherche initiale</li>
            <li>Juin: Implémentation des modèles</li>
            <li>Juillet: Débogage et génération des résultats</li>
            <li>Août: Rédaction du document</li>
        </ul>
    ",
    description_title: "Évaluation efficace des fonctions à l'aide de l'apprentissage par renforcement",
    description: "
        <div class=\"text-lighttext-800 dark:text-darktext-200\">
            <p>
                En sciences, l'évaluation de fonctions complexes, souvent s'étendant sur plusieurs dimensions, présente un défi computationnel significatif. Ces fonctions peuvent être coûteuses et prendre beaucoup de temps à évaluer avec précision. Le projet propose le développement d'une approche novatrice pour relever ce défi en tirant profit des réseaux neuronaux et des techniques d'apprentissage par renforcement pour émuler efficacement l'évaluation de fonctions arbitrairement complexes.
            </p>

            <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Objectifs</h3>
            <p>
                L'objectif principal du projet de recherche est de former un réseau neuronal capable d'émuler l'évaluation de fonctions complexes en une fraction du temps nécessaire pour évaluer directement la vraie fonction. Plus précisément, le projet vise à :
            </p>
            <ul>
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

            <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Méthodologie</h3>
            <p>La recherche utilisera une approche en deux étapes :</p>
            <ol>
                <li>
                    <strong>Apprentissage par Renforcement pour un Échantillonnage Efficace : </strong>
                    Un modèle d'apprentissage par renforcement sera formé pour échantillonner efficacement des points dans l'espace des fonctions. L'objectif du modèle sera d'apprendre à sélectionner des points qui contribuent le plus à représenter avec précision la fonction tout en minimisant le nombre d'évaluations nécessaires. Le modèle d'apprentissage par renforcement utilisera une fonction non différentiable comme signal de récompense, représentant la disparité entre la vraie fonction et son émulation. Cette fonction guidera le modèle pour échantillonner des points efficacement.
                </li>
                <li>
                    <strong>Émulation par Réseau Neuronal : </strong>
                    Un réseau neuronal sera formé en utilisant les points échantillonnés générés par le modèle d'apprentissage par renforcement. Ce réseau neuronal apprendra à approximer la fonction complexe en utilisant les points de données échantillonnés. En émulant la fonction, le réseau neuronal permettra une évaluation rapide de la fonction à n'importe quelle entrée donnée, réduisant ainsi considérablement les frais informatiques par rapport à une évaluation directe.
                </li>
            </ol>

            <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Résultats Attendus</h3>
            <ul>
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

            <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">
                Signification et Applications Potentielles
            </h3>
            <p>
                L'évaluation efficace des fonctions est cruciale dans diverses disciplines scientifiques, notamment la physique, l'ingénierie et l'apprentissage automatique. L'approche proposée a le potentiel de révolutionner les méthodes computationnelles en permettant une évaluation plus rapide et plus efficace de fonctions complexes. Les applications incluent les problèmes d'optimisation, l'analyse basée sur la simulation et la formation de modèles dans divers domaines.
            </p>

            <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Calendrier et Ressources</h3>
            <p>
                La recherche sera menée sur une période définie, en utilisant des ressources informatiques pour la formation et la validation du modèle. La collaboration avec des experts en apprentissage automatique, en optimisation et dans des domaines spécifiques enrichira le processus de recherche et garantira son succès.
            </p>

            <h3 class=\"text-lightaccent-600 dark:text-darkaccent-400\">Conclusion</h3>
            <p>
                Le projet propose une approche novatrice pour relever les défis computationnels associés à l'évaluation efficace de fonctions complexes. En combinant des réseaux neuronaux et des techniques d'apprentissage par renforcement, la recherche vise à développer une solution évolutive applicable à un large éventail de domaines scientifiques et d'ingénierie. Cette approche innovante a le potentiel d'accélérer les progrès en sciences computationnelles et de permettre des percées dans la résolution de problèmes complexes.
            </p>
        </div>
    ",
};
