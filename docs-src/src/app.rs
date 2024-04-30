use leptos::*;
use leptos_meta::*;

use crate::pages::{
    description::Description, planning::Planning, progress::Progress, sidebar::Sidebar,
};

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();
    view! {
        <div class="grid gap-5 grid-cols-1 lg:grid-cols-5 min-h-screen">
            <Sidebar/>
            <div class="flex flex-col flex-auto mx-auto lg:col-span-4 w-full lg:w-11/12">
                <div class="flex flex-col flex-auto gap-3 pb-8 lg:py-8 lg:h-0 lg:overflow-y-scroll lg:no-scrollbar">
                    <Description/>
                    <Planning/>
                    <Progress/>
                </div>
            </div>
        </div>
    }
}
