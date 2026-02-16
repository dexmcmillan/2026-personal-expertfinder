from scrapers.utils import clean_html, save_urls_to_csv, load_urls_from_csv


def test_clean_html_strips_nav_and_scripts():
    html = """
    <html>
    <head><script>var x = 1;</script></head>
    <body>
    <nav><a href="/">Home</a></nav>
    <header>Site Header</header>
    <main><p>Professor Jane Smith studies AI.</p></main>
    <footer>Copyright 2024</footer>
    </body>
    </html>
    """
    cleaned = clean_html(html)
    assert "Professor Jane Smith studies AI." in cleaned
    assert "var x = 1" not in cleaned
    assert "Site Header" not in cleaned
    assert "Copyright" not in cleaned


def test_save_and_load_urls(tmp_path):
    urls = [
        ("University of Waterloo", "Faculty of Engineering", "https://example.com/prof1"),
        ("University of Waterloo", "Faculty of Science", "https://example.com/prof2"),
    ]
    csv_path = tmp_path / "urls.csv"
    save_urls_to_csv(urls, csv_path)
    loaded = load_urls_from_csv(csv_path)
    assert len(loaded) == 2
    assert loaded[0] == {
        "school": "University of Waterloo",
        "faculty": "Faculty of Engineering",
        "url": "https://example.com/prof1",
    }
