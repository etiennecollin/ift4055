use leptos::*;
use leptos_meta::provide_meta_context;
use leptos_router::*;

use crate::{
    locale::{Languages, Locale, SiteQueries},
    pages::{description::Description, planning::Planning, progress::Progress, sidebar::Sidebar},
};

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    let (locale, locale_set) = create_signal(Locale::new(Languages::English));

    // This produces a blank page...
    // let query = use_query_map();
    // let lang = move || {
    //     query
    //         .with(|query| query.get("lang").cloned())
    //         .unwrap_or("en".to_owned())
    // };

    let on_click = move |_| {
        locale_set.set(Locale::new({
            if locale.get().lang == Languages::French {
                Languages::English
            } else {
                Languages::French
            }
        }))
    };

    view! {
        <div class="grid gap-5 grid-cols-1 lg:grid-cols-5 min-h-screen">
            <Sidebar locale=locale/>
            <div class="flex flex-col flex-auto mx-auto lg:col-span-4 w-full lg:w-11/12">
                <div class="flex flex-col flex-auto gap-3 pb-8 lg:py-8 lg:h-0 lg:overflow-y-scroll lg:no-scrollbar">
                    <button
                        class="text-lighttext-700 hover:text-lightaccent-600 dark:text-darktext-300 hover:dark:text-darkaccent-400 bg-lightbg-200 dark:bg-darkbg-800 rounded-xl py-2 px-3 lg:py-3 lg:px-5 w-fit items-middle"
                        on:click=on_click
                        inner_html=move || locale.get().get_lang_button_text()
                    ></button>
                    <a class="text-lighttext-700 hover:text-lightaccent-600 dark:text-darktext-300 hover:dark:text-darkaccent-400 bg-lightbg-200 dark:bg-darkbg-800 rounded-xl py-2 px-3 lg:py-3 lg:px-5 w-fit items-middle" href="/?lang=fr" inner_html=move || locale.get().get_lang_button_text()></a>
                    <Description locale=locale/>
                    <Planning locale=locale/>
                    <Progress locale=locale/>
                </div>
            </div>
        </div>
    }
}
