use ift4055_docs::app::App;
use leptos::*;
use leptos_meta::*;

fn main() {
    provide_meta_context();
    leptos::mount_to_body(|| {
        view! {
            <main class="mx-auto px-12 bg-lightbg-100 dark:bg-darkbg-900">
                <App/>
            </main>
        }
    });
}
