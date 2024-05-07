use crate::{locale::Locale, utils::section::Section};
use leptos::*;

#[component]
pub fn Planning(#[prop(into)] locale: ReadSignal<Locale>) -> impl IntoView {
    view! {
        <Section title="Planning">
            <div inner_html=move || locale.get().get_planning()></div>
        </Section>
    }
}
