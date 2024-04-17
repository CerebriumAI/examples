from cerebrium import __version__ as cerebrium_version
from cerebrium.commands.app import app  # noqa: E402 F401
from cerebrium.commands.auth import login, save_auth_config  # Import the login function
from cerebrium.commands.config import config_app
from cerebrium.commands.cortex import deploy, init
from cerebrium.commands.project import project_app
from cerebrium.commands.serve import serve_app  # noqa: E402 F401
from cerebrium.commands.storage import storage_app  # noqa: E402 F401
from cerebrium.core import cli  # noqa: E402 F401

cli.add_typer(storage_app, name="storage", help="Manage all storage operations")
cli.add_typer(
    app,
    name="app",
    help="Manage your apps. See a list of your apps, their details and scale them",
)
cli.add_typer(
    project_app, name="project", help="Manage all functionality around your projects"
)
cli.add_typer(
    config_app,
    name="config",
    help="Utility to manage your cerebrium config.",
)
cli.add_typer(
    serve_app,
    name="serve",
    help="Run code on a remote cortex server with hot reloading for rapid development",
)


@cli.command()
def version():
    """
    Print the version of the Cerebrium CLI
    """
    print(cerebrium_version)


# Add commands directly to the CLI
cli.command()(login)
cli.command()(save_auth_config)
cli.command()(init)
cli.command()(deploy)

if __name__ == "__main__":
    cli()
