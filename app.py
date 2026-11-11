
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Sultan Sharif Ali Learning Portal",
    page_icon=":material/school:",
    layout="wide"
)

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: #14532D;
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    color: #14532D;
    font-size: 32px;
    font-weight: 700;
}

.welcome-box {
    background-color: #14532D;
    padding: 30px;
    border-radius: 12px;
    color: white;
    margin-bottom: 25px;
}

.welcome-box h1 {
    color: white;
}

.welcome-box p {
    color: #DCFCE7;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
}

.top-clock {
    background-color: #006B3C;
    color: white;
    padding: 10px 18px;
    border-radius: 8px;
    white-space: nowrap;
    font-size: 14px;
}

/* University logo: not clickable, not draggable, not selectable */
.st-key-sidebar_logo,
.st-key-sidebar_logo * {
    user-select: none;
    -webkit-user-select: none;
    -webkit-touch-callout: none;
}

.st-key-sidebar_logo img {
    pointer-events: none;
    -webkit-user-drag: none;
    -moz-user-drag: none;
    -webkit-user-select: none;
    user-select: none;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# UNIVERSITY FOOTER WATERMARK
# ==========================================

st.markdown("""
<style>

.footer-watermark {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #006B3C;
    color: #FFFFFF;
    text-align: center;
    padding: 12px 0;
    font-size: 13px;
    font-weight: 500;
    border-top: 1px solid #168447;
    z-index: 999;
}

</style>

<div class="footer-watermark">
    Sultan Sharif Ali Islamic University
    © 2026 (01/12/2026)
</div>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "data" / "processed_resources.csv")


df = load_data()

# ==========================================
# 4. PREPARE TF-IDF SEARCH ENGINE
# ==========================================

documents = df["searchable_text"].fillna("")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),
    max_df=1.0,
    min_df=1
)

tfidf_matrix = vectorizer.fit_transform(documents)

# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    # University Logo (centred, not clickable/downloadable)
    with st.container(horizontal_alignment="center", key="sidebar_logo"):
        st.image(
            str(BASE_DIR / "assets" / "university_logo.png"),
            width=120
        )

    st.markdown("### University Sultan Sharif Ali")
    st.markdown("STUDENT LEARNING RESOURCES")

    st.space("stretch")

    st.divider()


    st.caption("This work is still in progress")
    st.caption("Any suggestion on the feature will be discussed during meeting")


# ==========================================
# 6. TOP NAVIGATION BAR (LEFT)
# ==========================================

def go_to_category(topic):
    st.session_state["browse_topic"] = topic
    st.session_state["top_navigation"] = "Browse Categories"


def go_back():
    if st.session_state["nav_history"]:
        previous_page = st.session_state["nav_history"].pop()
        st.session_state["going_back"] = True
        st.session_state["top_navigation"] = previous_page


NAV_ITEMS = {
    "Dashboard": ":material/dashboard: Dashboard",
    "Find Resources": ":material/search: Find Resources",
    "Browse Categories": ":material/category: Browse Categories",
    "About": ":material/info: About"
}

if "top_navigation" not in st.session_state:
    st.session_state["top_navigation"] = "Dashboard"

if "nav_history" not in st.session_state:
    st.session_state["nav_history"] = []

if "last_page" not in st.session_state:
    st.session_state["last_page"] = "Dashboard"

with st.container(
    horizontal=True,
    vertical_alignment="center",
    horizontal_alignment="left",
    gap="medium"
):

    page = st.segmented_control(
        "Navigation",
        options=list(NAV_ITEMS.keys()),
        format_func=lambda option: NAV_ITEMS[option],
        required=True,
        label_visibility="collapsed",
        key="top_navigation"
    )

    st.space("stretch")

    brunei_time = datetime.now(ZoneInfo("Asia/Brunei"))

    current_date = brunei_time.strftime("%d/%m/%Y")
    current_time = brunei_time.strftime("%I:%M %p")

    st.markdown(
        f'<div class="top-clock">{current_date} | {current_time}</div>',
        unsafe_allow_html=True,
        width="content"
    )

# Track navigation history so a back button can return to the previous page
if page != st.session_state["last_page"]:
    if not st.session_state.pop("going_back", False):
        st.session_state["nav_history"].append(st.session_state["last_page"])
        st.session_state["nav_history"] = st.session_state["nav_history"][-20:]
    st.session_state["last_page"] = page

if st.session_state["nav_history"]:
    st.button(
        f":material/arrow_back: Back to {st.session_state['nav_history'][-1]}",
        on_click=go_back,
        key="back_button"
    )

# Related learning categories shown when "Include related categories" is on
RELATED_TOPICS = {
    "Artificial Intelligence": [
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Data Mining",
        "Data Science"
    ],
    "Machine Learning": [
        "Artificial Intelligence",
        "Deep Learning",
        "Natural Language Processing",
        "Data Science",
        "Data Mining",
        "Statistics"
    ],
    "Deep Learning": [
        "Machine Learning",
        "Artificial Intelligence",
        "Natural Language Processing",
        "Data Science"
    ],
    "Natural Language Processing": [
        "Deep Learning",
        "Machine Learning",
        "Artificial Intelligence"
    ],
    "Data Science": [
        "Machine Learning",
        "Artificial Intelligence",
        "Statistics",
        "Data Mining",
        "Data Visualization",
        "Big Data Analytics"
    ],
    "Data Mining": [
        "Machine Learning",
        "Data Science",
        "Statistics",
        "Big Data Analytics",
        "Database Management"
    ],
    "Database Management": [
        "SQL",
        "Big Data Analytics",
        "Software Engineering"
    ],
    "SQL": [
        "Database Management",
        "Data Visualization",
        "Data Science"
    ],
    "Big Data Analytics": [
        "Data Mining",
        "Data Science",
        "Database Management",
        "Cloud Computing",
        "Statistics"
    ],
    "Data Visualization": [
        "Data Science",
        "Statistics",
        "SQL",
        "Big Data Analytics"
    ],
    "Statistics": [
        "Data Science",
        "Machine Learning",
        "Data Mining",
        "Big Data Analytics",
        "Data Visualization"
    ],
    "Python Programming": [
        "Software Engineering",
        "Data Science",
        "Machine Learning"
    ],
    "Cloud Computing": [
        "Big Data Analytics",
        "Database Management",
        "Software Engineering",
        "Cybersecurity"
    ],
    "Cybersecurity": [
        "Cloud Computing",
        "Software Engineering"
    ],
    "Software Engineering": [
        "Python Programming",
        "Cloud Computing",
        "Database Management",
        "Cybersecurity"
    ],
}



# ==========================================
# 7. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.markdown(
        '<div class="welcome-box">'
        '<h1>Welcome to University Islam Sultan Sharif Ali Learning Resource!</h1>'
        '<p>Discover resources, explore knowledge, '
        'and develop your skills.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="main-title">Dashboard</p>',
        unsafe_allow_html=True
    )

    total_resources = len(df)
    total_categories = df["topic"].nunique()
    total_types = df["resource_type"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Resources",
        total_resources
    )

    col2.metric(
        "Learning Categories",
        total_categories
    )

    col3.metric(
        "Resource Types",
        total_types
    )

    st.divider()

    st.subheader("Explore Learning Resources")

    st.write(
        "Use the navigation menu to search for "
        "educational materials and explore different topics."
    )

    st.subheader("Available Learning Categories")

    topics = sorted(df["topic"].dropna().unique())

    columns = st.columns(3)

    for index, topic in enumerate(topics):

        with columns[index % 3]:

            st.button(
                topic,
                key=f"category_button_{index}",
                width="stretch",
                on_click=go_to_category,
                args=(topic,)
            )

# ==========================================
# 8. FIND RESOURCES
# ==========================================

elif page == "Find Resources":

    st.markdown(
        '<p class="main-title">Find Learning Resources</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Search for tutorials, workshops, seminars "
        "and educational materials."
    )

    query = st.text_input(
        "What would you like to learn?",
        placeholder="Example: Python programming"
    )

    if st.button(":material/search: Search", type="primary"):

        if query.strip():

            query_vector = vectorizer.transform([query])

            scores = cosine_similarity(
                query_vector,
                tfidf_matrix
            ).flatten()

            ranked_indices = scores.argsort()[::-1]

            matches = [
                i for i in ranked_indices
                if scores[i] > 0
            ][:5]

            if not matches:

                st.warning("No matching resources found.")

            else:

                st.subheader("Search Results")

                st.success(
                    f"Found {len(matches)} matching resources."
                )

                for rank, index in enumerate(matches, start=1):

                    row = df.iloc[index]

                    with st.container(border=True):

                        st.markdown(
                            f"### {rank}. {row['title']}"
                        )

                        st.write(f"**Topic:** {row['topic']}")

                        st.write(
                            f"**Resource Type:** "
                            f"{row['resource_type']}"
                        )

                        st.write(row["summary"])

                        st.caption(
                            f"Similarity Score: "
                            f"{scores[index]:.4f}"
                        )

        else:

            st.warning("Please enter a search query.")

# ==========================================
# 9. BROWSE CATEGORIES
# ==========================================

elif page == "Browse Categories":

    st.markdown(
        '<p class="main-title">Browse Categories</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Explore educational resources by learning topic."
    )

    topics = sorted(df["topic"].dropna().unique())

    selected_topic = st.selectbox(
        "Select a learning category",
        topics,
        key="browse_topic"
    )

    filter_col1, filter_col2 = st.columns(2)

    resource_types = sorted(df["resource_type"].dropna().unique())

    with filter_col1:

        selected_types = st.multiselect(
            "Filter by resource type",
            resource_types,
            default=resource_types
        )

    with filter_col2:

        min_date = pd.to_datetime(df["date"]).min().date()
        max_date = pd.to_datetime(df["date"]).max().date()

        date_range = st.date_input(
            "Filter by date",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    include_related = st.toggle(
        "Include related categories",
        value=True,
        help="Also show resources from topics closely related to the selected category.",
        key="include_related"
    )

    related_topics = (
        RELATED_TOPICS.get(selected_topic, [])
        if include_related
        else []
    )

    shown_topics = [selected_topic] + related_topics

    # Exact-match resources first, then related categories in order
    category_df = pd.concat(
        [df[df["topic"] == topic] for topic in shown_topics]
    )

    # Apply resource-type and date filters
    dates = (
        date_range
        if isinstance(date_range, (list, tuple))
        else (date_range,)
    )

    start_date, end_date = dates[0], dates[-1]

    resource_dates = pd.to_datetime(category_df["date"]).dt.date

    filtered_df = category_df[
        category_df["resource_type"].isin(selected_types)
        & (resource_dates >= start_date)
        & (resource_dates <= end_date)
    ]

    exact_count = int((filtered_df["topic"] == selected_topic).sum())
    related_count = len(filtered_df) - exact_count

    st.subheader(selected_topic)

    if related_topics:
        st.write(
            f"Available resources: {exact_count} from {selected_topic} "
            f"+ {related_count} from related categories"
        )
        st.caption(
            f"Related categories included: {', '.join(related_topics)}"
        )
    else:
        st.write(
            f"Available resources: {exact_count}"
        )

    if len(filtered_df) != len(category_df):

        st.caption(
            f"Showing {len(filtered_df)} of {len(category_df)} "
            f"resources after applying filters."
        )

    if not selected_types:

        st.warning(
            "Select at least one resource type to see results."
        )

    elif len(filtered_df) == 0:

        st.info("No resources match the selected filters.")

    for _, row in filtered_df.iterrows():

        with st.container(border=True):

            st.markdown(f"### {row['title']}")

            st.write(f"**Topic:** {row['topic']}")

            st.write(
                f"**Resource Type:** {row['resource_type']}"
            )

            st.write(f"**Date:** {row['date']}")

            st.write(row["summary"])

# ==========================================
# 10. ABOUT
# ==========================================

elif page == "About":

    st.markdown(
        '<p class="main-title">About This Project</p>',
        unsafe_allow_html=True
    )

    st.write("""
    ### Learning Resource Finder

    Learning Resource Finder is an educational search
    application designed to help students discover
    relevant learning materials.

    ### Search Technology

    - Python
    - Streamlit
    - TF-IDF Vectorization
    - Cosine Similarity
    - Pandas
    - Scikit-learn

    ### Dataset

    This project uses a synthetic dataset containing
    150 educational resources.

    ### Purpose

    To make searching and discovering educational
    resources easier for university students.
    """)
