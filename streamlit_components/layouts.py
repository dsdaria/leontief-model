import streamlit as st

def info_box(text: str, box_type: str = "info"):
    box_class = "info-box"
    if box_type == "warning":
        box_class += " warning"
    elif box_type == "success":
        box_class += " success"
    
    st.markdown(f"""
    <div class="{box_class}">
        {text}
    </div>
    """, unsafe_allow_html=True)

def section_title(title: str, emoji: str = ""):
    emoji_span = f'<span class="emoji">{emoji}</span>' if emoji else ''
    
    st.markdown(f"""
    <div style="margin: 1rem 0 1.5rem 0;">
        <h2 style="color: #1e3c72; border-left: 4px solid #2a5298; padding-left: 1rem;">
            {emoji_span} {title}
        </h2>
    </div>
    """, unsafe_allow_html=True)