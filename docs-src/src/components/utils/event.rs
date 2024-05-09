use leptos::*;

use crate::utils::subsection::Subsection;

#[component]
pub fn Event(
    #[prop(into)] title: String,
    #[prop(into)] subtitle: String,
    #[prop(into)] date: String,
    #[prop(optional, into)] description: Option<String>,
    #[prop(optional, into)] list: Option<Vec<String>>,
) -> impl IntoView {
    let (description_signal, _) = create_signal(description);
    let (list_signal, _) = create_signal(list);

    view! {
        <Subsection>
            <div>
                <div class="flex justify-between">
                    <h3 class="text-lighttext-800 dark:text-darktext-200">{title}</h3>
                    <p class="text-lighttext-700 dark:text-darktext-300">{date}</p>
                </div>
                <h4 class="text-lighttext-800 dark:text-darktext-200">{subtitle}</h4>
            </div>
            <Show when=move || { description_signal.get().is_some() }>
                <p class="text-lighttext-600 dark:text-darktext-400">{description_signal.get().unwrap()}</p>
            </Show>
            <Show when=move || { list_signal.get().is_some() }>
                <ul>
                    {list_signal
                        .get()
                        .unwrap()
                        .into_iter()
                        .map(|item| view! { <li class="text-lighttext-600 dark:text-darktext-400">{item}</li> })
                        .collect::<Vec<_>>()}
                </ul>
            </Show>
        </Subsection>
    }
}
