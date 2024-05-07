use crate::{locale::Locale, utils::social::Social};
use leptos::*;

#[component]
pub fn Sidebar(#[prop(into)] locale: ReadSignal<Locale>) -> impl IntoView {
    view! {
        <section class="my-8 lg:col-span-1 flex flex-col grow">
            <div class="flex flex-row justify-between items-center lg:hidden">
                <div>
                    <h1
                        class="text-5xl font-semibold text-lighttext-950 dark:text-darktext-50"
                        inner_html=move || locale.get().get_sidebar_title()
                    ></h1>
                    <h2 class="pt-2 text-xl text-lighttext-950 dark:text-darktext-50">Etienne Collin</h2>
                    <div>
                        <a
                            class="text-lighttext-900 dark:text-darktext-100 no-underline hover:underline"
                            href="mailto:collin.etienne.contact@gmail.com"
                        >
                            "collin.etienne.contact@gmail.com"
                        </a>
                        <br/>
                        <a
                            class="text-lighttext-900 dark:text-darktext-100 no-underline hover:underline"
                            href="mailto:etienne.collin@umontreal.ca"
                        >
                            "etienne.collin@umontreal.ca"
                        </a>
                    </div>
                    <p
                        class="pt-2 grow text-lighttext-900 dark:text-darktext-100"
                        inner_html=move || locale.get().get_sidebar_description()
                    ></p>
                </div>
                <img
                    class="rounded-full w-1/3 lg:w-auto m-4 mb-2 lg:m-4 hover:animate-spin"
                    src="assets/images/profile.jpg"
                    alt="Profile picture"
                />
            </div>
            <div class="hidden justify-between items-center lg:block">
                <h1
                    class="text-5xl font-semibold text-lighttext-950 dark:text-darktext-50"
                    inner_html=move || locale.get().get_sidebar_title()
                ></h1>
                <img
                    class="rounded-full w-1/3 lg:w-auto m-4 mb-2 lg:m-4 hover:animate-spin"
                    src="assets/images/profile.jpg"
                    alt="Profile picture"
                />
                <h2 class="pt-2 text-xl text-lighttext-950 dark:text-darktext-50">Etienne Collin</h2>
                <div>
                    <a
                        class="text-lighttext-900 dark:text-darktext-100 no-underline hover:underline"
                        href="mailto:collin.etienne.contact@gmail.com"
                    >
                        "collin.etienne.contact@gmail.com"
                    </a>
                    <br/>
                    <a
                        class="text-lighttext-900 dark:text-darktext-100 no-underline hover:underline"
                        href="mailto:etienne.collin@umontreal.ca"
                    >
                        "etienne.collin@umontreal.ca"
                    </a>
                </div>
                <p
                    class="pt-2 grow text-lighttext-900 dark:text-darktext-100"
                    inner_html=move || locale.get().get_sidebar_description()
                ></p>
            </div>
            <div class="flex flex-row gap-5 mt-4">
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
        </section>
    }
}
