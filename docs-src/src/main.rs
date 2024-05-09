use ift4055_docs::app::App;
use leptos::*;
use leptos_meta::*;
use leptos_router::*;

fn main() {
    provide_meta_context();
    leptos::mount_to_body(|| {
        view! {
            <Router>
                <main class="mx-auto px-8 bg-lightbg-100 dark:bg-darkbg-900">
                    <App/>
                </main>
            </Router>
        }
    });
}
