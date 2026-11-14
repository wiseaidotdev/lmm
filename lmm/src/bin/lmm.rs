#[cfg(feature = "cli")]
use lmm::tui::run_tui;

use lmm::app::run_cli_entry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    #[cfg(feature = "cli")]
    if args.len() == 1 {
        return run_tui().await;
    }

    run_cli_entry(args).await
}
