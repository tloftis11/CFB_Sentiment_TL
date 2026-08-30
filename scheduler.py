"""
Daily scheduler — keeps rankings updated automatically.

Run once and leave it running:
    python scheduler.py

It will:
  - Run the pipeline immediately on startup
  - Re-run every day at the configured time (default 06:00 local)
  - Log failures without crashing the scheduler

For production on Render, this becomes the process command.
"""

import logging
import os
import sys
import time
from datetime import datetime

import schedule
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

RUN_TIME = os.getenv("PIPELINE_RUN_TIME", "06:00")  # HH:MM local time


def job():
    logger.info(f"--- Scheduled pipeline run starting at {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")
    try:
        from main import run_pipeline
        run_pipeline()
        logger.info("Scheduled run completed successfully.")
    except SystemExit:
        # run_pipeline calls sys.exit(1) on missing API key — propagate
        logger.error("Pipeline exited. Check CFBD_API_KEY in your .env file.")
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        logger.info("Scheduler will retry tomorrow.")


def main():
    logger.info(f"CFB Sentiment Scheduler started.")
    logger.info(f"Pipeline will run daily at {RUN_TIME} local time.")

    # Run immediately on startup so you get fresh data right away
    logger.info("Running initial pipeline now...")
    job()

    schedule.every().day.at(RUN_TIME).do(job)
    logger.info(f"Next scheduled run: {schedule.next_run()}")

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
