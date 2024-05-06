use crate::utils::section::Section;
use leptos::*;

#[component]
pub fn Planning() -> impl IntoView {
    view! {
        <Section title="Planning">
            <p class="text-lighttext-800 dark:text-darktext-200">
                "This entire project should span the 4 months of summer 2024. Here is a general outlook on the schedule."
            </p>
            <ul class="list-disc list-outside pl-8 lg:pl-6">
                <li>"May: Planning, reading and initial research"</li>
                <li>"June: Implementation of models"</li>
                <li>"July: Debugging and generation of results"</li>
                <li>"August: Writing for paper"</li>
            </ul>
        </Section>
    }
}
