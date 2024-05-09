use leptos::*;

#[component]
pub fn Social(
    #[prop(into)] href: String,
    #[prop(into)] label: String,
    #[prop(into)] fa_icon: String,
) -> impl IntoView {
    view! {
        <a href=href aria-label=label>
            <i class=format!(
                "{fa_icon} text-lighttext-700 hover:text-lightaccent-600 dark:text-darktext-300 hover:dark:text-darkaccent-400",
            )></i>
        </a>
    }
}
