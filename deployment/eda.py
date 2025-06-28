import streamlit as st

st.markdown("""
<style>
.custom-base {
    text-align: center;
    color: white;
    padding: 12px;
    border-radius: 8px;
    width: 100%;
    box-sizing: border-box;
}
.custom-markdown {
    background-color: #2c2f33;
    font-size: 16px;
}
.custom-title {
    background-color: transparent;
}
</style>
""", unsafe_allow_html=True)


def custom_md_clear(html):
    st.markdown(html, unsafe_allow_html=True)
    
# Custom styled markdown block
def custom_md(text):
    st.markdown(f"<div class='custom-base custom-markdow-clear'>{text}</div>", unsafe_allow_html=True)

# Custom styled title block
def custom_title(text):
    st.markdown(f"<h3 class='custom-base custom-title'>{text}</h3>", unsafe_allow_html=True)

def show():
    st.title("Traffic Sign Group Overview")
    custom_md_clear("""
                <p>This section introduces each group of traffic signs used in the classification model. Each group contains signs that are <b>visually and semantically related</b>, making it easier for the model to generalize and distinguish between types.</p>

                <p>The classification is structured according to the <b>Vienna Convention on Road Signs and Signals</b>, an internationally recognized system for traffic sign standards. Aligning the model with this convention ensures broader applicability in real-world transportation systems, especially in countries that adhere to these standards.</p>

                <p><b>Traffic Sign Groups:</b></p>
                <ul>
                    <li><b>Group 1: Speed Limit Signs</b> – Circular signs with white backgrounds and red borders indicating speed limits (e.g., 30 km/h, 50 km/h).</li>
                    <li><b>Group 2: Prohibitory Signs</b> – Restrictive signs such as <i>No Entry</i>, <i>No Overtaking</i>, or <i>No Trucks</i>, typically circular with red borders.</li>
                    <li><b>Group 3: Mandatory Signs</b> – Indicate required actions like <i>Turn Left</i>, <i>Go Straight</i>, or <i>Keep Right</i>, usually circular and blue.</li>
                    <li><b>Group 4: Warning Signs</b> – Triangular signs that alert drivers to hazards like <i>Curves</i>, <i>Bumps</i>, or <i>Slippery Roads</i>.</li>
                    <li><b>Group 5: Information Signs</b> – Square or rectangular signs providing helpful information such as <i>Parking</i>, <i>Priority Road</i>, or <i>End of Restrictions</i>.</li>
                </ul>
                """)

    st.markdown('---')
    for i in range(1, 6):
        group_image_path = f"Images/Group{i}.png"
        group_titles = {
            1: "Group 1: Speed Limit Signs",
            2: "Group 2: Prohibitory Signs",
            3: "Group 3: Mandatory Signs",
            4: "Group 4: Warning Signs",
            5: "Group 5: Information Signs"
        }
        group_descriptions = {
            1: "Includes all circular white signs with red borders indicating speed limits (e.g., 30, 50, 70 km/h).",
            2: "<li>All signs are circular with a red border.</li><li>The internal symbols are black, with many signs featuring a red slash to indicate a forbidden action.</li>",
            3: "Circular blue signs that indicate mandatory directions or actions (e.g., turn right, go straight).",
            4: "Triangular shape with a yellow or white background and black or red symbols, indicates a warning for the driver for what to come",
            5: "All signs are circular with a blue background and white symbols, associated with  informational signs"
        }

        custom_title(group_titles[i])
        left, center, right = st.columns([0.5, 2, 0.5])
        with center:
            st.image(group_image_path, use_container_width=True)
        custom_md(group_descriptions[i])
        st.markdown("---")