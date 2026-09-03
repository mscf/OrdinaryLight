from pathlib import Path


def test_ordinarylight_does_not_import_ordinaryscience():
    root = Path(__file__).resolve().parents[1] / "ordinarylight"
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
    )
    assert "import ordinaryscience" not in sources
    assert "from ordinaryscience" not in sources
