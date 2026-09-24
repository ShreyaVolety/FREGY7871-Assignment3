"""Command-line orchestrator for the news and market pipeline."""
import argparse
import subprocess
import sys


def run(module):
    print(f"\n>>> {module}", flush=True)
    subprocess.run([sys.executable, "-m", module], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-news-download", action="store_true",
                        help="Reuse data/raw/articles_raw.csv")
    parser.add_argument("--after-review", action="store_true",
                        help="Promote reviewed candidates and run H/L + market analysis")
    args = parser.parse_args()
    if not args.skip_news_download:
        run("src.collect_news")
    run("src.process_news")
    run("src.score_news_days")
    if not args.after_review:
        print("\nSTOP: review data/processed/candidate_high_news_days.csv, then rerun with --after-review")
        return
    run("src.promote_news_candidates")
    run("src.build_event_pairs")
    run("src.download_market_data")
    run("src.run_analysis")


if __name__ == "__main__":
    main()
