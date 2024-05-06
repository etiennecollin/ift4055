use leptos::*;

#[component]
pub fn SectionSwitch(
    #[prop(into)] title_1: String,
    #[prop(into)] title_2: String,
    #[prop(optional)] extra_classes: Option<String>,
    children: ChildrenFn,
) -> impl IntoView {
    let (classes_signal, _) = create_signal(extra_classes);
    let class_string = format!("bg-lightbg-200 dark:bg-darkbg-800 rounded-xl py-2 px-3 lg:py-3 lg:px-5 {} print:break-before-all print:break-after-all print:break-inside-avoid", classes_signal.get().unwrap_or_default());
    let show = create_rw_signal(true);
    let children = store_value(children);
    let (title_1_signal, _) = create_signal(title_1);
    let (title_2_signal, _) = create_signal(title_2);

    view! {
        <section class=class_string>
            <div class="flex justify-between items-center">
                <h2 class="text-xl text-lightaccent-600 dark:text-darkaccent-400">
                    <Show when=move || show.get()>{title_1_signal.get()}</Show>
                    <Show when=move || !show.get()>{title_2_signal.get()}</Show>
                </h2>
                <button
                    class="text-xl text-lighttext-700 hover:text-lightaccent-600 dark:text-darktext-300 hover:dark:text-darkaccent-400"
                    on:click=move |_| show.update(|x| *x = !*x)
                >
                    <a aria-label="Change Language">
                        <Show when=move || show.get()>"FR"</Show>
                        <Show when=move || !show.get()>"EN"</Show>
                        <i class=format!("fa-solid fa-arrow-right")></i>
                    </a>
                </button>
            </div>

            <div class="grid gap-2 pt-2">
                <Show when=move || {
                    show.get()
                }>{children.with_value(|children| children().nodes.into_iter().nth(0))}</Show>
                <Show when=move || {
                    !show.get()
                }>{children.with_value(|children| children().nodes.into_iter().nth(1))}</Show>
            </div>
        </section>
    }
}
