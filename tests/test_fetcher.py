from internet_explorer.fetcher import detect_render_profile


def test_detect_render_profile_static() -> None:
    html = "<html><head><title>A</title></head><body><h1>Data Portal</h1><p>" + ("content " * 400) + "</p></body></html>"
    assert detect_render_profile(html) == "static_ssr"


def test_detect_render_profile_csr_shell() -> None:
    html = (
        "<html><body><div id='root'></div>"
        + ("<script>bundle()</script>" * 15)
        + "</body></html>"
    )
    assert detect_render_profile(html) == "csr_shell"
