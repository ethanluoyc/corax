import nox


@nox.session
def test(session):
    session.install("-r", "requirements/test.txt", "jax[cpu]", "-e", ".[tf,jax]")
    session.run("pytest", "-n", "auto", "corax/")


@nox.session
def lint(session):
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files")
