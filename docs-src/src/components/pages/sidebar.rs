use crate::{locale::Locale, utils::social::Social};
use leptos::*;
use leptos_router::*;

#[component]
pub fn Sidebar() -> impl IntoView {
    let locale =
        use_context::<ReadSignal<Locale>>().expect("expecting locale signal to be provided");

    let (small, small_set) = create_signal(String::new());
    let (large, large_set) = create_signal(String::new());

    create_effect(move |_| {
        let title = format!(
            "<h1 class=\"font-semibold text-lighttext-950 dark:text-darktext-50\">{}</h1>",
            locale.get().sidebar_title
        );
        let subtitle = "<h2 class=\"pt-2 text-lighttext-950 dark:text-darktext-50\">Etienne Collin</h2>";
        // let contact = "<div><a class=\"text-lighttext-900 dark:text-darktext-100 no-underline hover:underline\" href=\"mailto:collin.etienne.contact@gmail.com\">collin.etienne.contact@gmail.com</a><br/><a class=\"text-lighttext-900 dark:text-darktext-100 no-underline hover:underline\" href=\"mailto:etienne.collin@umontreal.ca\">etienne.collin@umontreal.ca</a></div>";
        let description = format!(
            "<p class=\"pt-2 grow text-lighttext-900 dark:text-darktext-100\">{}</p>",
            locale.get().sidebar_description
        );
        let image = "<img class=\"rounded-full w-1/3 lg:w-auto m-4 mb-2 lg:m-4 hover:animate-spin\" src=\"assets/images/profile.jpg\" alt=\"Profile picture\"/>";

        small_set.set(format!(
            "<div>{}{}{}</div>{}",
            title, subtitle,description, image
        ));
        large_set.set(format!(
            "{}{}{}{}",
            title, image, subtitle, description
        ));
    });

    view! {
        <section class="my-8 lg:col-span-1 flex flex-col grow">
            // <div class="flex flex-row justify-between items-center lg:hidden">
            <div class="flex flex-row justify-between items-center lg:hidden" inner_html=move || small.get()></div>
            <div class="hidden justify-between items-center lg:block" inner_html=move || large.get()></div>
            <div class="flex flex-row gap-5 my-4">
                <Social
                    href="https://github.com/etiennecollin"
                    label="Checkout my GitHub"
                    fa_icon="fa-brands fa-github"
                />
                <Social
                    href="https://www.linkedin.com/in/etiennecollin"
                    label="Checkout my LinkedIn"
                    fa_icon="fa-brands fa-linkedin"
                />
                <Social href="https://etiennecollin.com" label="Checkout my website" fa_icon="fa-solid fa-globe"/>
            </div>
            <div class="flex flex-row gap-2 mt-auto">
                <A
                    class="text-lighttext-700 hover:text-lightaccent-600 dark:text-darktext-300 hover:dark:text-darkaccent-400 bg-lightbg-200 dark:bg-darkbg-800 rounded-xl py-2 px-3 lg:py-3 lg:px-5 w-fit items-middle"
                    href="?lang=en"
                >
                    "EN"
                </A>
                <A
                    class="text-lighttext-700 hover:text-lightaccent-600 dark:text-darktext-300 hover:dark:text-darkaccent-400 bg-lightbg-200 dark:bg-darkbg-800 rounded-xl py-2 px-3 lg:py-3 lg:px-5 w-fit items-middle"
                    href="?lang=fr"
                >
                    "FR"
                </A>
            </div>
        </section>
    }
}
