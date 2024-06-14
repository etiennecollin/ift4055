use crate::{
    locale::Locale,
    utils::{event::Event, section::Section},
};
use leptos::*;

#[component]
pub fn Progress() -> impl IntoView {
    let locale =
        use_context::<ReadSignal<Locale>>().expect("expecting locale signal to be provided");

    let (title, title_set) = create_signal(String::new());
    let (progress_1_title, progress_1_title_set) = create_signal(String::new());
    let (progress_1_description, progress_1_description_set) = create_signal(String::new());
    let (progress_2_title, progress_2_title_set) = create_signal(String::new());
    let (progress_2_description, progress_2_description_set) = create_signal(String::new());
    let (progress_3_title, progress_3_title_set) = create_signal(String::new());
    let (progress_3_description, progress_3_description_set) = create_signal(String::new());

    create_effect(move |_| {
        title_set.set(locale.get().progress_title.to_owned());
        progress_1_title_set.set(locale.get().progress_1_title.to_owned());
        progress_1_description_set.set(locale.get().progress_1_description.to_owned());
        progress_2_title_set.set(locale.get().progress_2_title.to_owned());
        progress_2_description_set.set(locale.get().progress_2_description.to_owned());
        progress_3_title_set.set(locale.get().progress_3_title.to_owned());
        progress_3_description_set.set(locale.get().progress_3_description.to_owned());
    });

    view! {
        <Section title=title>
            <Event
                title=progress_1_title
                date="2024/05/01 - 2024/05/14"
                description=progress_1_description
                list=vec![
                    "<a href=\"https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf\">Stanford RL textbook</a>"
                        .to_owned(),
                    "<a href=\"https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html\">PyTorch TorchRL PPO tutorial</a>"
                        .to_owned(),
                    "<a href=\"https://arxiv.org/abs/2402.04355\">PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation</a>"
                        .to_owned(),
                    "<a href=\"https://www.sciencedirect.com/science/article/pii/S2666827022000640\">Intelligent sampling for surrogate modeling, hyperparameter optimization, and data analysis</a>"
                        .to_owned(),
                    "<a href=\"https://openreview.net/pdf?id=ThMSXaolvn\">Multi-Fidelity Active Learning with GFlowNets</a>"
                        .to_owned(),
                ]
            />

            <Event
                title=progress_2_title
                date="2024/05/14 - 2024/05/28"
                description=progress_2_description
                list=vec![
                    "<a href=\"https://github.com/pytorch/examples/blob/37a1866d0e0118875d52071756f76b9b3e46c565/reinforcement_learning/actor_critic.py\">PyTorch examples - actor-critic</a>"
                        .to_owned(),
                    "<a href=\"https://github.com/yc930401/Actor-Critic-pytorch/tree/master\">DQN to play Cartpole game with pytorch</a>"
                        .to_owned(),
                    "<a href=\"https://youtube.com/playlist?list=PLvSH07QabjqZRKKuq92HN7zXqUIDk6Nyx&si=62VRmB9NDBT-AKB5\">MILA GFlowNets Workshop 2023</a>"
                        .to_owned(),
                    "<a href=\"https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3\">The GFlowNet Tutorial</a>"
                        .to_owned(),
                    "<a href=\"https://github.com/GFNOrg/torchgfn/\">torchgfn Python package repository</a>".to_owned(),
                ]
            />

            <Event
                title=progress_3_title
                date="2024/05/28 - 2024/06/11"
                description=progress_3_description
                list=vec![
                    "<a href=\"https://www.youtube.com/watch?v=ttE0F7fghfk\">Hyperparameter Optimization - The Math of Intelligence #7</a>"
                        .to_owned(),
                    "<a href=\"https://www.youtube.com/watch?v=UBDgSHPxVME&t=620s\">Gaussian Processes</a>"
                        .to_owned(),
                    "<a href=\"https://www.youtube.com/watch?v=M-NTkxfd7-8&t=4s\">Bayesian Optimization (Bayes Opt): Easy explanation of popular hyperparameter tuning method</a>"
                        .to_owned(),
                    "<a href=\"https://www.youtube.com/watch?v=5Cqi-RAwAu8\">I get confused trying to learn Gaussian Processes | Learn with me!</a>"
                        .to_owned(),
                    "<a href=\"https://www.youtube.com/watch?v=iDzaoEwd0N0\">Easy introduction to gaussian process regression (uncertainty models)</a>"
                        .to_owned(),
                ]
            />
        </Section>
    }
}
