use crate::{
    locale::Locale,
    utils::{event::Event, section::Section},
};
use leptos::*;

#[component]
pub fn Progress(#[prop(into)] locale: ReadSignal<Locale>) -> impl IntoView {
    view! {
        <Section title=locale.get().get_progress_title()>
            <Event
                title="Title"
                subtitle="Subtitle"
                date="May 1st 2024 - May 3rd 2024"
                description="Progress description".to_owned()
                list=vec!["Task list item 1".to_owned(), "Task list item 2".to_owned()]
            />

        </Section>
    }
}
