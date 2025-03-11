import streamlit as st
from multimodal_search import MultiModalSearch
import time

# Set the page config
st.set_page_config(
    layout="wide",
    page_title="Fashion Cloth Search App",
    page_icon="👗"
)

def main():
    # Add the heading
    st.markdown("<h1 style='text-align: center; color: green;'>Fashion Cloth Search App</h1>", unsafe_allow_html=True)
    
    # Add a sidebar for configuration
    with st.sidebar:
        st.header("Search Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)
        confidence_threshold = st.slider("Confidence threshold (%)", min_value=0, max_value=100, value=0)
    
    # Initialized multimodal search
    multimodal_search = MultiModalSearch()

    # Add tabs for different search modes
    tab1, tab2 = st.tabs(["Text Search", "Image Search"])
    
    with tab1:
        # prompt user for entering query
        query = st.text_input("Enter your text query:")
        if st.button("Search", key="text_search"):
            if len(query) > 0:
                with st.spinner("Searching..."):
                    # call the search function
                    results = multimodal_search.search(query, top_k=top_k)
                
                st.info(f"Showing top {len(results)} results for: '{query}'")
                
                # Display results in a grid
                display_results(results, confidence_threshold/100)
            else:
                st.warning("Please enter a query.")
    
    with tab2:
        # Image upload for search
        uploaded_image = st.file_uploader("Upload an image to search similar items", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", width=200)
            if st.button("Search Similar", key="image_search"):
                st.info("Image search functionality will be implemented in the future.")
                # Future implementation: multimodal_search.search_by_image(uploaded_image, top_k=top_k)

def display_results(results, threshold=0):
    # Filter results based on confidence threshold
    filtered_results = [r for r in results if r.score >= threshold]
    
    if not filtered_results:
        st.warning(f"No results found with confidence above {threshold*100}%")
        return
        
    # Create a dynamic grid based on number of results
    cols = st.columns(min(3, len(filtered_results)))
    
    for i, result in enumerate(filtered_results):
        col_idx = i % len(cols)
        with cols[col_idx]:
            confidence = round(result.score*100, 2)
            st.metric("Confidence", f"{confidence}%")
            st.image(result.content, use_column_width=True)
            
            # Extract filename for display
            filename = os.path.basename(result.content)
            st.caption(f"File: {filename}")

if __name__ == "__main__":
    import os
    main()