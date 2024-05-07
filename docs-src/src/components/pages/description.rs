use crate::{locale::Locale, utils::section::Section};
use leptos::*;

#[component]
pub fn Description(locale: ReadSignal<Locale>) -> impl IntoView {
    view! {
        <div>
            <Section title=locale.get().get_description_title()>
                <div
                    class="text-lighttext-800 dark:text-darktext-200"
                    inner_html=move || locale.get().get_description()
                ></div>
            </Section>
        </div>
    }
}
