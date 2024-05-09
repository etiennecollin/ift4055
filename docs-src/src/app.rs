use leptos::*;
use leptos_meta::provide_meta_context;
use leptos_router::*;

use crate::{
    locale::{LOCALE_EN, LOCALE_FR},
    pages::{description::Description, planning::Planning, progress::Progress, sidebar::Sidebar},
};

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    let (locale, locale_set) = create_signal(LOCALE_EN);
    provide_context(locale);

    let query = use_query_map();

    let lang = move || query.with(|query| query.get("lang").cloned().unwrap_or_default());

    create_effect(move |_| match lang().as_str() {
        "en" => locale_set.set(LOCALE_EN),
        "fr" => locale_set.set(LOCALE_FR),
        _ => locale_set.set(LOCALE_EN),
    });

    view! {
        <div class="grid lg:gap-5 grid-cols-1 lg:grid-cols-5 min-h-screen">
            <Sidebar/>
            <Routes>
                <Route path="/" view=AppContainer>
                    <Route path="" view=AppContent/>
                </Route>
            </Routes>
        </div>
    }
}

#[component]
fn AppContainer() -> impl IntoView {
    view! {
        <div class="flex flex-col flex-auto mx-auto lg:col-span-4 w-full lg:w-11/12">
            <div class="flex flex-col flex-auto gap-3 pb-8 lg:py-8 lg:h-0 lg:overflow-y-scroll lg:no-scrollbar">
                <Outlet/>
            </div>
        </div>
    }
}

#[component]
fn AppContent() -> impl IntoView {
    view! {
        <Description/>
        <Planning/>
        <Progress/>
    }
}
