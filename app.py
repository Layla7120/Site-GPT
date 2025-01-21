import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain.document_transformers import Html2TextTransformer

st.set_page_config(
    page_title="Site GPT",
    page_icon="🖥️",
)

html2text_transformer = Html2TextTransformer()

def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
    )

@st.cache_data(show_spinner="Loading websites...")
def load_website(url):
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ]
    )
    loader.requests_per_second = 5
    docs = loader.load()
    return docs

st.markdown(
    """
    # SiteGPT

    Ask questions about the content of a website.

    Start by writing the URL of the website on the sidebar.
    """
)

with st.sidebar:
    url = st.text_input(
        "Enter a valid sitemap to inquire about.",
        value="https://developers.cloudflare.com/sitemap.xml",
        placeholder="https://example.com/sitemap.xml",
    )

if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please enter a Sitemap URL.")
    else:
        docs = load_website(url)
        st.write(docs)

