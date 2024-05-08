use leptos::*;

#[component]
pub fn Subsection(
    #[prop(optional)] extra_classes: Option<String>,
    children: Children,
) -> impl IntoView {
    let class_string = format!("bg-lightbg-300 dark:bg-darkbg-700 rounded-xl py-2 px-3 {} print:break-before-all print:break-after-all print:break-inside-avoid", extra_classes.unwrap_or_default());
    view! { <section class=class_string>{children().nodes}</section> }
}
