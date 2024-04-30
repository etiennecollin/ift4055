use crate::utils::social::Social;
use leptos::*;

#[component]
pub fn Sidebar() -> impl IntoView {
    let title = "<h1 class=\"text-5xl font-semibold text-lighttext-950 dark:text-darktext-50\">Hi, I'm Etienne</h1>";
    let subtitle = "<h2 class=\"pt-2 text-xl text-lighttext-950 dark:text-darktext-50\">Student & Schulich Leader</h2>";
    let contact = "<a class=\"text-lighttext-900 dark:text-darktext-100 no-underline hover:underline\" href=\"mailto:collin.etienne.contact@gmail.com\">collin.etienne.contact@gmail.com</a>";
    let description =
        "<p class=\"pt-2 grow text-lighttext-900 dark:text-darktext-100\">Built with love using Rust 🦀 and NeoVim 🖥️</p>";
    let image = "<img class=\"rounded-full w-1/3 lg:w-auto m-4 mb-2 lg:m-4 hover:animate-spin\" src=\"assets/images/profile.jpg\" alt=\"Profile picture\"/>";

    let small = format!(
        "<div>{}{}{}{}</div>{}",
        title, subtitle, contact, description, image
    );
    let large = format!("{}{}{}{}{}", title, image, subtitle, contact, description);

    view! {
        <section class="my-8 lg:col-span-1 flex flex-col grow">
            <div class="flex flex-row justify-between items-center lg:hidden" inner_html=small></div>
            <div class="hidden justify-between items-center lg:block" inner_html=large></div>
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
