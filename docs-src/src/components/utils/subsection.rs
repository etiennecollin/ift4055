use leptos::*;

#[component]
pub fn Subsection(
    #[prop(optional)] extra_classes: Option<String>,
    children: Children,
) -> impl IntoView {
    let (classes_signal, _) = create_signal(extra_classes);
    let class_string = format!("bg-lightbg-300 dark:bg-darkbg-700 rounded-xl py-2 px-3 {} print:break-before-all print:break-after-all print:break-inside-avoid", classes_signal.get().unwrap_or_default());
    view! { <section class=class_string>{children().nodes}</section> }
}
