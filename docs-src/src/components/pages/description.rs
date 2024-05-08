use crate::{locale::Locale, utils::section::Section};
use leptos::*;

#[component]
pub fn Description() -> impl IntoView {
    let locale =
        use_context::<ReadSignal<Locale>>().expect("expecting locale signal to be provided");

    let (title, title_set) = create_signal(String::new());
    create_effect(move |_| {
        title_set.set(locale.get().description_title.to_owned());
    });

    view! {
        <div>
            <Section title=title>
                <div
                    class="text-lighttext-800 dark:text-darktext-200"
                    inner_html=move || locale.get().description
                ></div>
            </Section>
        </div>
    }
}
