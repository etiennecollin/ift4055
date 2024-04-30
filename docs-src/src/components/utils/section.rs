use leptos::*;

#[component]
pub fn Section(
    #[prop(into)] title: String,
    #[prop(optional)] extra_classes: Option<String>,
    children: Children,
) -> impl IntoView {
    let (classes_signal, _) = create_signal(extra_classes);
    let class_string = format!("bg-lightbg-200 dark:bg-darkbg-800 rounded-xl py-2 px-3 lg:py-3 lg:px-5 {} print:break-before-all print:break-after-all print:break-inside-avoid", classes_signal.get().unwrap_or_default());
    view! {
        <section class=class_string>
            <h2 class="text-xl text-lightaccent-600 dark:text-darkaccent-400">{title}</h2>
            <div class="grid gap-2 pt-2">{children().nodes}</div>
        </section>
    }
}
