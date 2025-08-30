import click
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--force', is_flag=True, help='Force reprocessing of photos')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def main(path: str, force: bool, verbose: bool):
    """Process photos from PATH (file or directory)."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Processing photos from: {path}")
    
    # TODO: Implement processing logic
    click.echo(f"Processing: {path}")
    if force:
        click.echo("Force mode enabled")

if __name__ == "__main__":
    main()