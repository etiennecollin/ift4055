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
    create_effect(move |_| {
        title_set.set(locale.get().progress_title.to_owned());
    });

    view! {
        <Section title=title>
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
