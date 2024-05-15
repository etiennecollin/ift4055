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

    create_effect(move |_| {
        title_set.set(locale.get().progress_title.to_owned());
        progress_1_title_set.set(locale.get().progress_1_title.to_owned());
        progress_1_description_set.set(locale.get().progress_1_description.to_owned());
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
                    "<a href=\"https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html\"> PyTorch TorchRL PPO tutorial</a>"
                        .to_owned(),
                    "<a href=\"https://arxiv.org/abs/2402.04355\">PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation</a>"
                        .to_owned(),
                    "<a href=\"https://www.sciencedirect.com/science/article/pii/S2666827022000640\">Intelligent sampling for surrogate modeling, hyperparameter optimization, and data analysis</a>"
                        .to_owned(),
                    "<a href=\"https://openreview.net/pdf?id=ThMSXaolvn\">Multi-Fidelity Active Learning with GFlowNets</a>"
                        .to_owned(),
                ]
            />

        </Section>
                }
}
