import typer

app = typer.Typer(help="fmritool CLI")

@app.command()
def hello(name: str = "world"):
    """Say hello (sanity-check command)."""
    print(f"hello {name}")

if __name__ == "__main__":
    app()
