from internet_explorer.fetcher import analyze_page, detect_render_profile
from internet_explorer.models import FetchResult


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


def test_analyze_page_flags_cloudflare_style_challenge_as_captcha() -> None:
    html = (
        "<html><head><title>Just a moment...</title></head>"
        "<body><h1>Just a moment...</h1><p>Checking your browser before accessing the site.</p></body></html>"
    )
    evidence = analyze_page(
        FetchResult(
            url="https://example.com/",
            final_url="https://example.com/",
            status_code=403,
            content_type="text/html",
            html=html,
            body_text=html,
            text_excerpt="Just a moment... Checking your browser before accessing the site.",
        )
    )

    assert evidence.captcha_present is True
