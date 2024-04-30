use crate::utils::section::Section;
use leptos::*;

#[component]
pub fn Description() -> impl IntoView {
    view! {
        <Section title="Description">
            <p class="text-lighttext-800 dark:text-darktext-200">
                "Project description"
            </p>
        </Section>
    }
}
