# Website

This website for the IFT4055 class at Université de Montréal is written using Rust using the [Leptos full stack framework](https://github.com/leptos-rs/leptos) and [Tailwind CSS](https://github.com/tailwindlabs/tailwindcss).

## Dependencies

- [Rust](https://github.com/rust-lang/rust)
- The following rust target: `wasm32-unknown-unknown`
  - Install it with: `rustup target add wasm32-unknown-unknown`
- [Trunk](https://github.com/trunk-rs/trunk)
  - Install it with: `cargo install --locked trunk`

## Usage

Use the following command to open a live preview of the website:

```bash
trunk serve --open
```

To compile the website, run:

```bash
trunk serve --open
```

To automatically compile and deploy the website (you will only have to push a commit), run:

```bash
./release.sh
```
