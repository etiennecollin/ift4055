use crate::{locale::Locale, utils::section::Section};
use leptos::*;

#[component]
pub fn Planning() -> impl IntoView {
    let locale =
        use_context::<ReadSignal<Locale>>().expect("expecting locale signal to be provided");

    let (title, title_set) = create_signal(String::new());
    create_effect(move |_| {
        title_set.set(locale.get().planning_title.to_owned());
    });

    view! {
        <Section title=title>
            <div inner_html=move || locale.get().planning></div>
        </Section>
    }
}
