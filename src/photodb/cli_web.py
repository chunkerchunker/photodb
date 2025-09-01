import click
import logging
from .web import create_app
from .utils.logging import setup_logging
from dotenv import load_dotenv

load_dotenv()


@click.command()
@click.option("--port", default=5000, help="Port to run the web server on")
@click.option("--host", default="127.0.0.1", help="Host to bind the web server to")
@click.option("--debug/--no-debug", default=False, help="Run in debug mode")
def main(port: int, host: str, debug: bool):
    """Start the photo browsing web server."""
    logger = setup_logging(logging.INFO)
    logger.info(f"Starting PhotoDB web server on {host}:{port}")

    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
